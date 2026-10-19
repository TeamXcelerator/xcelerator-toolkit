//! Source-bound algebraic index check in the requested subspace.
//! Two directed inertia counts isolate index zero of the exact stored matrix
//! for Natural/AdaptiveEven, or its even compression for EvenSector. The latter
//! does not assert an ordering against the odd block or the unrounded CCM form.
use super::{state_residual_bounds, stored_resolution, CcmParityPolicy};
use crate::ccm::capture_runtime::CaptureResourcePolicy;
use crate::ccm::retained_evidence::finite_math::scale_float;
use anyhow::{bail, Result};
use rug::{float::Round, Float};
#[cfg(test)]
use xc_certify::exact::interval_symmetric_ldlt_inertia_mpfr_stable;
use xc_certify::{exact::IntervalInertiaResult, CertificateError};
#[cfg(test)]
use xc_numerics::interval::RationalInterval;
use xc_numerics::{
    interval::IntervalError,
    mpfr_interval::MpfrInterval as I,
    symmetric_inertia::{self, MpfrInertiaResult},
};

pub(super) const ARITHMETIC: &str =
    "directed_stored_source_stable_ground_index_and_rounding_scale_gap_v3";

/// Outcome of the directed index proof for well-formed inputs within budget.
/// Invalid inputs and exceeded budgets are errors, never outcomes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Outcome {
    Proven,
    /// The proof failed or stayed inconclusive; the reason is deterministic.
    NotEstablished(String),
}

std::thread_local! {
    pub(super) static CHECKED_SOURCES: std::cell::RefCell<Vec<xc_cache::ContentDigest>> = const { std::cell::RefCell::new(Vec::new()) };
    /// Sources whose proof did not succeed, with the reason, so that a
    /// producer and its admission replay count the inertia once.
    static UNESTABLISHED_SOURCES: std::cell::RefCell<Vec<(xc_cache::ContentDigest, String)>> = const { std::cell::RefCell::new(Vec::new()) };
    #[cfg(test)]
    pub(super) static FULL_GROUND_CHECKS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    #[cfg(test)]
    static PIVOT_RECORD_CONVERSIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
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
        let lower = xc_numerics::fmt::decimal_string_round(
            &Float::with_val_round(work, pivot.lower(), Round::Down).0,
            Some(16),
            Round::Down,
        );
        let upper = xc_numerics::fmt::decimal_string_round(
            &Float::with_val_round(work, pivot.upper(), Round::Up).0,
            Some(16),
            Round::Up,
        );
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

#[cfg(test)]
fn admit_rational_workspace(
    matrix: &[I],
    p: u32,
    retained_source_bytes: u128,
    maximum_bytes: u128,
) -> Result<()> {
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
    check_workspace(bytes, maximum_bytes, "rational inertia")
}

#[cfg(test)]
fn rational_matrix(
    matrix: &[I],
    p: u32,
    retained_source_bytes: u128,
    maximum_bytes: u128,
) -> Result<Vec<RationalInterval>> {
    admit_rational_workspace(matrix, p, retained_source_bytes, maximum_bytes)?;
    Ok(matrix.iter().map(I::to_rational_interval).collect())
}

// A finite p-bit endpoint survives the exact dyadic rational round trip
// unchanged, except that Rational has no negative zero. Preserve that
// normalization without constructing an O(n^2) rational image.
fn canonicalize_rational_zeros(matrix: &mut [I]) {
    for value in matrix {
        let positive_zero = |x: &Float| {
            if x.is_zero() {
                Float::with_val(x.prec(), 0)
            } else {
                x.clone()
            }
        };
        if (value.lower().is_zero() && value.lower().is_sign_negative())
            || (value.upper().is_zero() && value.upper().is_sign_negative())
        {
            *value = I::new(positive_zero(value.lower()), positive_zero(value.upper()))
                .expect("canonicalizing a validated zero preserves its interval");
        }
    }
}

#[cfg(test)]
fn direct_inertia(matrix: &[I], n: usize, p: u32) -> Result<IntervalInertiaResult> {
    Ok(direct_inertia_bounded(matrix, n, p, 0, u128::MAX)?.value)
}

struct AdmittedInertia {
    value: IntervalInertiaResult,
    rational_bytes: u128,
    _reservation: Option<xc_core::MemoryReservation>,
}

// Elimination can produce exponents absent from the source matrix. Inspect
// the actual O(d) pivots before any exact dyadic integer allocation.
fn pivot_record_bytes(values: &[I]) -> Result<u128> {
    let mut total = 0u128;
    let mut largest = 0u128;
    for value in values {
        value.validate()?;
        for endpoint in [value.lower(), value.upper()] {
            let bits = 2 * u128::from(endpoint.prec())
                + u128::from(i64::from(endpoint.get_exp().unwrap_or(0)).unsigned_abs())
                + 2;
            total = total.checked_add(bits).ok_or_else(workspace_overflow)?;
            largest = largest.max(bits);
        }
    }
    // Numerator+denominator storage, headers, current conversion scratch,
    // and conservative GMP capacity/temporary allowance. No dense rationals.
    total
        .div_ceil(8)
        .checked_add(4 * largest.div_ceil(8))
        .and_then(|bytes| bytes.checked_add(values.len() as u128 * 256))
        .and_then(|bytes| bytes.checked_mul(12))
        .ok_or_else(workspace_overflow)
}

fn direct_inertia_bounded(
    matrix: &[I],
    n: usize,
    p: u32,
    already_live_bytes: u128,
    maximum_bytes: u128,
) -> Result<AdmittedInertia> {
    let run = || {
        // Retain the certificate adapter's boundary errors even though the
        // production caller constructs a symmetric matrix of known shape.
        if !(32..=1_000_000).contains(&p) || n == 0 || n.checked_mul(n) != Some(matrix.len()) {
            return Err(IntervalError::Invalid(
                "invalid interval inertia shape or precision".into(),
            ));
        }
        for row in 0..n {
            for column in 0..row {
                let a = &matrix[row * n + column];
                let b = &matrix[column * n + row];
                if a.lower() != b.lower() || a.upper() != b.upper() {
                    return Err(IntervalError::Invalid(
                        "asymmetric rational inertia matrix".into(),
                    ));
                }
            }
        }
        symmetric_inertia::inertia_stable(matrix, n, p)
    };
    let result = run().map_err(|error| {
        CertificateError::VerificationFailed(format!("MPFR stable interval inertia: {error}"))
    })?;
    let values = match &result {
        MpfrInertiaResult::Conclusive {
            pivot_enclosures, ..
        }
        | MpfrInertiaResult::Inconclusive {
            pivot_enclosures, ..
        } => pivot_enclosures,
    };
    let rational_bytes = pivot_record_bytes(values)?;
    check_workspace(
        already_live_bytes
            .checked_add(rational_bytes)
            .ok_or_else(workspace_overflow)?,
        maximum_bytes,
        "pivot records",
    )?;
    let reservation = crate::ccm::capture_runtime::reserve_working(rational_bytes)?;
    let records = |values: Vec<I>| {
        #[cfg(test)]
        PIVOT_RECORD_CONVERSIONS.with(|count| count.set(count.get() + values.len()));
        values.iter().map(I::to_rational_interval).collect()
    };
    let value = match result {
        MpfrInertiaResult::Conclusive {
            positive,
            negative,
            pivot_enclosures,
        } => IntervalInertiaResult::Conclusive {
            positive,
            negative,
            pivot_enclosures: records(pivot_enclosures),
        },
        MpfrInertiaResult::Inconclusive {
            pivot_index,
            positive,
            negative,
            zero_or_unresolved,
            pivot_enclosures,
            reason,
        } => IntervalInertiaResult::Inconclusive {
            pivot_index,
            positive,
            negative,
            zero_or_unresolved,
            pivot_enclosures: records(pivot_enclosures),
            reason,
        },
    };
    Ok(AdmittedInertia {
        value,
        rational_bytes,
        _reservation: reservation,
    })
}

/// Reject impossible dimension/precision allocations before constructing Tau.
/// Actual pivot exponent spans are admitted separately before exact conversion.
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

fn workspace_overflow() -> anyhow::Error {
    anyhow::anyhow!("ground-index workspace estimate overflow")
}

fn scalar_bytes(p: u32) -> u128 {
    u128::from(p).div_ceil(64) * 8 + 160
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
    let (n, d) = (n as u128, d as u128);
    let work = p.checked_add(1024).ok_or_else(workspace_overflow)?;
    let residual_work = p.checked_add(64).ok_or_else(workspace_overflow)?;
    let solver_work = p
        .checked_mul(2)
        .and_then(|p| p.checked_add(64))
        .ok_or_else(workspace_overflow)?
        .min(1_000_000);
    // Known producer/replay caller ownership: Tau and an extra source image;
    // sector and ordinary/guarded retained LU images, plus spare dense capacity.
    // Rc LU factors are not RunResources-owned. All four d^2 images are charged
    // at the larger solver precision. Existing run reservations remain additive.
    let caller = 2 * n * n * scalar_bytes(p)
        + 4 * d * d * scalar_bytes(solver_work)
        + (32 * n + 256) * scalar_bytes(solver_work)
        + 65_536;
    let normalized = n * n * scalar_bytes(p);
    // The optional residual calculation finishes before shifted inertia.
    // Four endpoint slots per full cell conservatively cover its actual two
    // endpoint MPFR values plus simultaneous Result<I>/I collection storage.
    let residual = (4 * n * n + 64 * n + 512) * scalar_bytes(residual_work);
    // Shifted matrix and inertia's mutable clone: four endpoints per cell.
    // O(d) allowance covers two pivot columns, records, support Vecs, and all
    // task-local Schur scratch (at most d row tasks, not merely pool workers).
    let inertia = (4 * d * d + 96 * d + 1024) * scalar_bytes(work);
    // Portable eigenstate input/output/round-trip coexistence is linear in n;
    // the cold producer emits finite p-bit values with compact exponents.
    let reports = 32 * n * (u128::from(p).div_ceil(3) + 128);
    caller
        .checked_add(normalized)
        .and_then(|bytes| bytes.checked_add(residual.max(inertia)))
        .and_then(|bytes| bytes.checked_add(reports))
        .ok_or_else(workspace_overflow)
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
) -> Result<Vec<I>> {
    let dimension = if even { n / 2 + 1 } else { n };
    // Validate before allocating even the first shifted interval. The outer
    // proof reservation also covers the source normalization and caller state.
    check_workspace(
        (a.len() as u128
            + 4 * dimension as u128 * dimension as u128
            + 96 * dimension as u128
            + 1024)
            * scalar_bytes(p),
        maximum_bytes,
        "shifted interval inertia",
    )?;
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
    canonicalize_rational_zeros(&mut matrix);
    Ok(matrix)
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
    match outcome_with_residual(a, v, lambda, p, parity, residual)? {
        Outcome::Proven => Ok(()),
        Outcome::NotEstablished(reason) => bail!("{reason}"),
    }
}

/// The index proof of [`validate_with_residual`] as an outcome: a failed or
/// inconclusive proof is [`Outcome::NotEstablished`] with the reason that
/// `validate_with_residual` reports; invalid inputs and budgets stay errors.
pub(super) fn outcome_with_residual(
    a: &[Float],
    v: &[Float],
    lambda: &Float,
    p: u32,
    parity: CcmParityPolicy,
    residual: Option<&state_residual_bounds::ResidualBounds>,
) -> Result<Outcome> {
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
        return Ok(Outcome::Proven);
    }
    if let Some(reason) = UNESTABLISHED_SOURCES.with(|cache| {
        cache
            .borrow()
            .iter()
            .find(|(key, _)| *key == digest)
            .map(|(_, reason)| reason.clone())
    }) {
        return Ok(Outcome::NotEstablished(reason));
    }
    let outcome = validate_unretained(a, v, lambda, p, parity, residual)?;
    match &outcome {
        Outcome::Proven => CHECKED_SOURCES.with(|cache| {
            let mut cache = cache.borrow_mut();
            if cache.len() >= 8 {
                cache.remove(0);
            }
            cache.push(digest);
        }),
        Outcome::NotEstablished(reason) => UNESTABLISHED_SOURCES.with(|cache| {
            let mut cache = cache.borrow_mut();
            if cache.len() >= 8 {
                cache.remove(0);
            }
            cache.push((digest, reason.clone()));
        }),
    }
    Ok(outcome)
}

fn validate_unretained(
    a: &[Float],
    v: &[Float],
    lambda: &Float,
    p: u32,
    parity: CcmParityPolicy,
    residual: Option<&state_residual_bounds::ResidualBounds>,
) -> Result<Outcome> {
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
    let fixed_bytes = workspace_bytes(n, p, parity)?;
    // Retain admission through residual/bounds, all serial guard rungs, and
    // all live rational result vectors. No arithmetic or proof is skipped.
    let _fixed_reservation = crate::ccm::capture_runtime::reserve_working(fixed_bytes)?;
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
            direct_inertia_bounded(&lower_matrix, dimension, work, fixed_bytes, maximum_bytes)?;
        drop(lower_matrix);
        match lower_result.value {
            IntervalInertiaResult::Conclusive { negative: 0, .. } => {}
            IntervalInertiaResult::Conclusive { .. } => return Ok(Outcome::NotEstablished(
                "claimed CCM eigenpair is not the algebraic ground state in the requested subspace"
                    .into(),
            )),
            ref result @ IntervalInertiaResult::Inconclusive { .. } => {
                let detail = inconclusive_summary(result, "lower", work);
                xc_core::progress_message!("[HP] CCM ground-index inertia: {detail}");
                inconclusive.push(detail);
                continue;
            }
        }
        let upper_matrix = shifted_matrix(&normalized, n, &upper, even, work, maximum_bytes)?;
        let upper_live = fixed_bytes
            .checked_add(lower_result.rational_bytes)
            .ok_or_else(workspace_overflow)?;
        let upper_result =
            direct_inertia_bounded(&upper_matrix, dimension, work, upper_live, maximum_bytes)?;
        match upper_result.value {
            IntervalInertiaResult::Conclusive { negative: 1, .. } => {
                if dimension == 1 {
                    return Ok(Outcome::Proven);
                }
                drop(upper_matrix);
                let neighbor_matrix =
                    shifted_matrix(&normalized, n, &neighbor_probe, even, work, maximum_bytes)?;
                let neighbor_live = upper_live.checked_add(upper_result.rational_bytes)
                    .ok_or_else(workspace_overflow)?;
                let neighbor_result = direct_inertia_bounded(
                    &neighbor_matrix, dimension, work, neighbor_live, maximum_bytes,
                )?;
                match neighbor_result.value {
                    IntervalInertiaResult::Conclusive { negative: 1, .. } => return Ok(Outcome::Proven),
                    IntervalInertiaResult::Conclusive { .. } => return Ok(Outcome::NotEstablished(format!(
                        "CCM stored precision resolution limit: selected-subspace gap is not separated from the matrix rounding scale at {p} bits"))),
                    ref result @ IntervalInertiaResult::Inconclusive { .. } => {
                        let detail = inconclusive_summary(result, "neighbor", work);
                        xc_core::progress_message!("[HP] CCM ground-index inertia: {detail}");
                        inconclusive.push(detail);
                        continue;
                    }
                }
            }
            IntervalInertiaResult::Conclusive { negative, .. } => {
                return Ok(Outcome::NotEstablished(format!(
                    "CCM ground state is unresolved or clustered at the requested precision: upper_count={negative}, eigenvalue={lambda}, residual={}",
                    bounds.residual_upper()
                )))
            }
            ref result @ IntervalInertiaResult::Inconclusive { .. } => {
                let detail = inconclusive_summary(result, "upper", work);
                xc_core::progress_message!("[HP] CCM ground-index inertia: {detail}");
                inconclusive.push(detail);
            }
        }
    }
    Ok(Outcome::NotEstablished(format!("CCM ground-index inertia is inconclusive within the guard precision budget: eigenvalue={}, residual_upper={}; {}", xc_numerics::fmt::decimal_string_round(lambda, Some(16), Round::Nearest), bounds.residual_upper(), inconclusive.join("; "))))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Frozen allocating route: construct the same integer-reflection
    // congruence, then serialize every interval through exact rationals.
    fn reference_shifted_rationals(
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
        let retained_source_bytes = n as u128 * n as u128 * (u128::from(p).div_ceil(8) + 160) * 16;
        rational_matrix(&matrix, p, retained_source_bytes, maximum_bytes)
    }

    #[test]
    fn artifact_direct_ground_inertia_matches_rational_adapter() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        pool.install(|| {
            let mut conclusive = 0;
            let mut inconclusive = 0;
            for p in [64, 128, 512] {
                for exponent in [-5000, 0, 5000] {
                    for entries in [
                        [2, 0, 0, 0, 0, 0, 0, 0, 2],
                        [0, 1, 0, 1, 0, 1, 0, 1, 0],
                        [3, -1, 2, -1, 1, -1, 2, -1, 3],
                    ] {
                        let matrix = entries
                            .into_iter()
                            .enumerate()
                            .map(|(i, value)| {
                                let x: Float = Float::with_val(p + 37, value) << exponent;
                                if value == 0 && i % 2 == 0 {
                                    -x
                                } else {
                                    x
                                }
                            })
                            .collect::<Vec<_>>();
                        for even in [false, true] {
                            let dimension = if even { 2 } else { 3 };
                            for shift in [-1, 0, 1, 3] {
                                let shift = Float::with_val(p + 37, shift) << exponent;
                                let reference = reference_shifted_rationals(
                                    &matrix,
                                    3,
                                    &shift,
                                    even,
                                    p,
                                    8 << 30,
                                )
                                .unwrap();
                                let direct =
                                    shifted_matrix(&matrix, 3, &shift, even, p, 8 << 30).unwrap();
                                // Compare the actual MPFR input, including zero signs,
                                // before comparing every exact pivot and diagnostic.
                                for (actual, expected) in direct.iter().zip(&reference) {
                                    let lower =
                                        Float::with_val_round(p, expected.lower(), Round::Down).0;
                                    let upper =
                                        Float::with_val_round(p, expected.upper(), Round::Up).0;
                                    assert_eq!(
                                        actual.lower().total_cmp(&lower),
                                        std::cmp::Ordering::Equal
                                    );
                                    assert_eq!(
                                        actual.upper().total_cmp(&upper),
                                        std::cmp::Ordering::Equal
                                    );
                                }
                                let expected = interval_symmetric_ldlt_inertia_mpfr_stable(
                                    &reference, dimension, p,
                                )
                                .map_err(|e| e.to_string());
                                let actual = direct_inertia(&direct, dimension, p)
                                    .map_err(|e| e.to_string());
                                assert_eq!(actual, expected);
                                match actual {
                                    Ok(IntervalInertiaResult::Conclusive { .. }) => conclusive += 1,
                                    Ok(result @ IntervalInertiaResult::Inconclusive { .. }) => {
                                        inconclusive += 1;
                                        if let Ok(expected) = expected {
                                            assert_eq!(
                                                inconclusive_summary(&result, "upper", p),
                                                inconclusive_summary(&expected, "upper", p)
                                            );
                                        }
                                    }
                                    Err(_) => {}
                                }
                            }
                        }
                    }
                }
            }
            assert!(conclusive > 0 && inconclusive > 0);
        });
    }

    #[test]
    fn artifact_direct_ground_adapter_preserves_zero_and_admission_errors() {
        let p = 128;
        for (lower_negative, upper_negative) in
            [(false, false), (true, false), (false, true), (true, true)]
        {
            let zero = |negative| {
                if negative {
                    -Float::with_val(p, 0)
                } else {
                    Float::with_val(p, 0)
                }
            };
            let mut matrix = [I::new(zero(lower_negative), zero(upper_negative)).unwrap()];
            let reference = rational_matrix(&matrix, p, 0, 8 << 30).unwrap();
            canonicalize_rational_zeros(&mut matrix);
            assert!(!matrix[0].lower().is_sign_negative() && !matrix[0].upper().is_sign_negative());
            assert_eq!(
                direct_inertia(&matrix, 1, p).unwrap(),
                interval_symmetric_ldlt_inertia_mpfr_stable(&reference, 1, p).unwrap()
            );
            for (n, precision) in [(0, p), (2, p), (1, 1_000_001)] {
                assert_eq!(
                    direct_inertia(&matrix, n, precision)
                        .unwrap_err()
                        .to_string(),
                    interval_symmetric_ldlt_inertia_mpfr_stable(&reference, n, precision)
                        .unwrap_err()
                        .to_string(),
                );
            }
        }
        let big = Float::with_val(p, 1) << 700_000_000i32;
        let matrix = vec![I::point(big); 5];
        // Both paths refuse before a huge rational allocation, with exactly
        // the same estimate and error. The optimization does not widen admission.
        assert_eq!(
            admit_rational_workspace(&matrix, p, 0, 8 << 30)
                .unwrap_err()
                .to_string(),
            rational_matrix(&matrix, p, 0, 8 << 30)
                .unwrap_err()
                .to_string(),
        );
        let source = vec![Float::with_val(p, 0); 9];
        // Both remain fail-closed. The new production refusal charges the
        // actual shifted interval stage, not the removed dense rational image.
        assert!(shifted_matrix(&source, 3, &source[0], true, p, 1)
            .unwrap_err()
            .to_string()
            .contains("shifted interval inertia"));
        assert!(reference_shifted_rationals(&source, 3, &source[0], true, p, 1).is_err());
    }

    #[test]
    fn ground_proof_outcomes_reuse_only_identical_points_and_parity() {
        CHECKED_SOURCES.with(|cache| cache.borrow_mut().clear());
        UNESTABLISHED_SOURCES.with(|cache| cache.borrow_mut().clear());
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
        // A failed proof is remembered with its reason, like a successful one.
        let mut reasons = Vec::new();
        for _ in 0..2 {
            let error =
                validate(&a, &wrong_vector, &wrong_value, p, CcmParityPolicy::Natural).unwrap_err();
            reasons.push(error.to_string());
            assert_eq!(FULL_GROUND_CHECKS.with(|count| count.get()), 3);
        }
        assert_eq!(reasons[0], reasons[1]);
        let mut wrong_source = a.clone();
        wrong_source[0] = Float::with_val(p, 0);
        assert!(validate(&wrong_source, &v, &value, p, CcmParityPolicy::Natural).is_err());
        assert_eq!(FULL_GROUND_CHECKS.with(|count| count.get()), 4);
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
        // No dense rational image exists in production. The owned-storage
        // model admits this case, while the frozen allocating oracle below
        // still accounts for the rationals it really constructs.
        preflight_with_budget(1001, 1000, CcmParityPolicy::EvenSector, 8 << 30).unwrap();
        preflight_with_budget(301, 1725, CcmParityPolicy::EvenSector, 8 << 30).unwrap();
        // Former 64-Mbit cap rejected this modest rational image even for zeros.
        let entries = vec![I::from_i64(0, 1789); 151 * 151];
        rational_matrix(&entries, 1789, 301 * 301 * 512 * 16, 8 << 30).unwrap();
    }
    #[test]
    fn ground_index_admission_respects_declared_budget_at_campaign_dimensions() {
        for (n, bytes) in [
            (241, 326_083_864u128),
            (1001, 5_259_675_544),
            (1601, 13_345_592_344),
            (2001, 20_790_603_544),
        ] {
            assert_eq!(
                workspace_bytes(n, 3386, CcmParityPolicy::EvenSector).unwrap(),
                bytes
            );
            assert_eq!(
                preflight_with_budget(n, 3386, CcmParityPolicy::EvenSector, 8 << 30).is_ok(),
                n <= 1001
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
    fn ground_index_owned_workspace_planner_is_bounded_and_checked() {
        for parity in [
            CcmParityPolicy::EvenSector,
            CcmParityPolicy::Natural,
            CcmParityPolicy::AdaptiveEven,
        ] {
            for p in [64, 128, 3386, 1_000_000] {
                for n in [1, 3, 241, 1001, 1601, 2001, 8193] {
                    let bytes = workspace_bytes(n, p, parity).unwrap();
                    preflight_with_budget(n, p, parity, bytes).unwrap();
                    assert!(preflight_with_budget(n, p, parity, bytes - 1).is_err());
                }
            }
        }
        for (n, p) in [
            (0, 128),
            (2, 128),
            (8195, 128),
            (usize::MAX, 128),
            (3, 0),
            (3, 63),
            (3, 1_000_001),
            (3, u32::MAX),
        ] {
            assert!(workspace_bytes(n, p, CcmParityPolicy::EvenSector).is_err());
        }
        assert_eq!(
            workspace_bytes(2001, 3386, CcmParityPolicy::EvenSector).unwrap(),
            20_790_603_544
        );
        assert!(
            workspace_bytes(2001, 3386, CcmParityPolicy::Natural).unwrap()
                > workspace_bytes(2001, 3386, CcmParityPolicy::EvenSector).unwrap()
        );
    }

    #[test]
    fn ground_index_bounded_inertia_matches_high_budget_rational_oracle() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        pool.install(|| {
            let mut conclusive = 0;
            let mut inconclusive = 0;
            for base in [64, 128] {
                for guard in [64, 256, 1024] {
                    let p = base + guard;
                    for entries in [
                        [2, 0, 0, 0, 1, 0, 0, 0, 2],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 1, 0, 1, 0, 1, 0],
                    ] {
                        let source = entries
                            .into_iter()
                            .enumerate()
                            .map(|(j, v)| {
                                let x = Float::with_val(base, v);
                                if v == 0 && j % 2 == 0 {
                                    -x
                                } else {
                                    x
                                }
                            })
                            .collect::<Vec<_>>();
                        for even in [false, true] {
                            let dimension = if even { 2 } else { 3 };
                            for shift in [-1, 0, 1] {
                                let shift = Float::with_val(p, shift);
                                let original = reference_shifted_rationals(
                                    &source,
                                    3,
                                    &shift,
                                    even,
                                    p,
                                    8 << 30,
                                )
                                .unwrap();
                                let matrix =
                                    shifted_matrix(&source, 3, &shift, even, p, 8 << 20).unwrap();
                                for (actual, exact) in matrix.iter().zip(&original) {
                                    assert_eq!(
                                        actual.lower().total_cmp(
                                            &Float::with_val_round(p, exact.lower(), Round::Down).0
                                        ),
                                        std::cmp::Ordering::Equal
                                    );
                                    assert_eq!(
                                        actual.upper().total_cmp(
                                            &Float::with_val_round(p, exact.upper(), Round::Up).0
                                        ),
                                        std::cmp::Ordering::Equal
                                    );
                                }
                                let expected = interval_symmetric_ldlt_inertia_mpfr_stable(
                                    &original, dimension, p,
                                )
                                .unwrap();
                                let resources =
                                    xc_core::RunResources::with_worker_limit(8 << 20, 1).unwrap();
                                let _scope = resources.enter();
                                let fixed = resources.reserve_working(1 << 20).unwrap();
                                let actual =
                                    direct_inertia_bounded(&matrix, dimension, p, 1 << 20, 8 << 20)
                                        .unwrap();
                                assert_eq!(actual.value, expected);
                                match &actual.value {
                                    IntervalInertiaResult::Conclusive { .. } => conclusive += 1,
                                    result @ IntervalInertiaResult::Inconclusive { .. } => {
                                        inconclusive += 1;
                                        assert_eq!(
                                            inconclusive_summary(result, "lower", p),
                                            inconclusive_summary(&expected, "lower", p)
                                        );
                                    }
                                }
                                assert!(resources.usage().working_bytes > 1 << 20);
                                drop(actual);
                                assert_eq!(resources.usage().working_bytes, 1 << 20);
                                drop(fixed);
                                assert_eq!(resources.usage().working_bytes, 0);
                            }
                        }
                    }
                }
            }
            assert!(conclusive > 0 && inconclusive > 0);
        });
    }

    #[test]
    fn ground_index_pivot_record_budget_precedes_exact_conversion() {
        let p = 128;
        PIVOT_RECORD_CONVERSIONS.with(|count| count.set(0));
        let huge = I::point(Float::with_val(p, 1) << 700_000_000i32);
        let matrix = [huge];
        let bytes = pivot_record_bytes(&matrix).unwrap();
        assert!(bytes > 1 << 20);
        let error = match direct_inertia_bounded(&matrix, 1, p, 0, 1 << 20) {
            Ok(_) => panic!("extreme pivot unexpectedly admitted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("pivot records"));
        assert_eq!(PIVOT_RECORD_CONVERSIONS.with(|count| count.get()), 0);

        let zero = [I::from_i64(0, p)];
        let bytes = pivot_record_bytes(&zero).unwrap();
        let first = direct_inertia_bounded(&zero, 1, p, 0, bytes).unwrap();
        assert!(matches!(
            first.value,
            IntervalInertiaResult::Inconclusive { .. }
        ));
        assert!(direct_inertia_bounded(&zero, 1, p, 0, bytes - 1).is_err());
        assert!(direct_inertia_bounded(&zero, 1, p, u128::MAX, bytes).is_err());
        assert_eq!(PIVOT_RECORD_CONVERSIONS.with(|count| count.get()), 1);

        // Previous result storage remains live while the next stage is admitted.
        assert!(direct_inertia_bounded(&zero, 1, p, first.rational_bytes, bytes).is_err());
        let next = direct_inertia_bounded(&zero, 1, p, first.rational_bytes, 2 * bytes).unwrap();
        assert_eq!(first.value, next.value);
    }

    #[test]
    fn ground_index_owned_reservations_release_and_refusal_is_not_memoized() {
        CHECKED_SOURCES.with(|cache| cache.borrow_mut().clear());
        UNESTABLISHED_SOURCES.with(|cache| cache.borrow_mut().clear());
        let p = 128;
        let a = [2, 0, 0, 0, 1, 0, 0, 0, 2]
            .into_iter()
            .map(|v| Float::with_val(p, v))
            .collect::<Vec<_>>();
        let v = [0, 1, 0]
            .into_iter()
            .map(|v| Float::with_val(p, v))
            .collect::<Vec<_>>();
        let value = Float::with_val(p, 1);
        let resources = xc_core::RunResources::with_worker_limit(1, 1).unwrap();
        {
            let _scope = resources.enter();
            let error = validate(&a, &v, &value, p, CcmParityPolicy::EvenSector).unwrap_err();
            assert!(
                error
                    .to_string()
                    .starts_with("ground-index validation exceeds workspace budget:"),
                "{error:#}"
            );
            assert!(error.to_string().contains("maximum_working_bytes=1"));
        }
        assert_eq!(resources.usage().working_bytes, 0);
        assert!(CHECKED_SOURCES.with(|cache| cache.borrow().is_empty()));
        assert!(UNESTABLISHED_SOURCES.with(|cache| cache.borrow().is_empty()));
        // A sufficient configured budget can still be occupied by another
        // owner. This exercises the reservation refusal after preflight passes.
        let required =
            u64::try_from(workspace_bytes(3, p, CcmParityPolicy::EvenSector).unwrap()).unwrap();
        let resources = xc_core::RunResources::with_worker_limit(required, 1).unwrap();
        {
            let _scope = resources.enter();
            let held = resources.reserve_working(1).unwrap();
            let error = validate(&a, &v, &value, p, CcmParityPolicy::EvenSector).unwrap_err();
            assert!(
                error.to_string().contains("run memory budget exceeded"),
                "{error:#}"
            );
            assert_eq!(resources.usage().working_bytes, 1);
            drop(held);
        }
        assert_eq!(resources.usage().working_bytes, 0);
        assert!(CHECKED_SOURCES.with(|cache| cache.borrow().is_empty()));
        assert!(UNESTABLISHED_SOURCES.with(|cache| cache.borrow().is_empty()));
        let resources = xc_core::RunResources::with_worker_limit(16 << 20, 1).unwrap();
        {
            let _scope = resources.enter();
            validate(&a, &v, &value, p, CcmParityPolicy::EvenSector).unwrap();
        }
        assert_eq!(resources.usage().working_bytes, 0);
        assert_eq!(CHECKED_SOURCES.with(|cache| cache.borrow().len()), 1);

        let matrix = [I::from_i64(0, p)];
        let bytes = u64::try_from(pivot_record_bytes(&matrix).unwrap()).unwrap();
        let resources = xc_core::RunResources::with_worker_limit(bytes + 100, 1).unwrap();
        let _scope = resources.enter();
        let retained = resources.try_reserve_retained(1).unwrap();
        let fixed = resources.reserve_working(100).unwrap();
        PIVOT_RECORD_CONVERSIONS.with(|count| count.set(0));
        assert!(direct_inertia_bounded(&matrix, 1, p, 100, u128::from(bytes + 100)).is_err());
        assert_eq!(PIVOT_RECORD_CONVERSIONS.with(|count| count.get()), 0);
        assert_eq!(resources.usage().working_bytes, 100);
        drop(retained);
        let admitted = direct_inertia_bounded(&matrix, 1, p, 100, u128::from(bytes + 100)).unwrap();
        drop(admitted);
        drop(fixed);
        assert_eq!(resources.usage().working_bytes, 0);
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
