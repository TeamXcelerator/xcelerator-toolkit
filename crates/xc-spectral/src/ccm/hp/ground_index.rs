//! Source-bound algebraic index check in the requested subspace.
//! Two directed inertia counts isolate index zero of the exact stored matrix
//! for Natural/AdaptiveEven, or its even compression for EvenSector. The latter
//! does not assert an ordering against the odd block or the unrounded CCM form.
use super::{stored_resolution, CcmParityPolicy};
use crate::ccm::retained_evidence::finite_math::scale_float;
use anyhow::{bail, Result};
use rug::Float;
use xc_certify::exact::{interval_symmetric_ldlt_inertia_mpfr, IntervalInertiaResult};
use xc_numerics::{interval::RationalInterval, mpfr_interval::MpfrInterval as I};

pub(super) const ARITHMETIC: &str = "directed_stored_source_ground_index_and_rounding_scale_gap_v2";

fn rational_matrix(
    matrix: &[I],
    p: u32,
    retained_source_bytes: u128,
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
    if bytes > (8u128 << 30) {
        bail!("ground-index inertia exceeds rational workspace budget");
    }
    Ok(matrix.iter().map(I::to_rational_interval).collect())
}

/// Reject impossible dimension/precision allocations before constructing Tau.
/// Actual exponent spans remain subject to rational_matrix's byte accounting.
pub(super) fn preflight(n: usize, p: u32, parity: CcmParityPolicy) -> Result<()> {
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
    if numerical + rational_minimum > (8u128 << 30) {
        bail!("ground-index validation exceeds numerical workspace budget");
    }
    Ok(())
}

// Integer reflection basis: e_center and e_(center-k)+e_(center+k).
// Congruence of A-tI preserves inertia without approximating sqrt(2).
fn shifted_matrix(
    a: &[Float],
    n: usize,
    shift: &Float,
    even: bool,
    p: u32,
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
    rational_matrix(&matrix, p, retained_source_bytes)
}

pub(super) fn validate(
    a: &[Float],
    v: &[Float],
    lambda: &Float,
    p: u32,
    parity: CcmParityPolicy,
) -> Result<()> {
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
    preflight(n, p, parity)?;
    let even = parity == CcmParityPolicy::EvenSector;
    if even
        && (v.iter().zip(v.iter().rev()).any(|(x, y)| x != y)
            || a.iter().zip(a.iter().rev()).any(|(x, y)| x != y))
    {
        bail!("even ground-index source requires exact reflection symmetry");
    }
    let maximum_work = p.saturating_add(1024);
    if a.len() as u128 * (u128::from(maximum_work).div_ceil(8) + 160) * 16 > (8u128 << 30) {
        bail!("ground-index validation exceeds numerical workspace budget");
    }
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
    let bounds = stored_resolution::bounds(a, v, lambda, p, parity)?;
    let dimension = if even { n / 2 + 1 } else { n };
    for guard in [64, 256, 1024] {
        let work = p.saturating_add(guard);
        let lower = scale_float(&bounds.lower, -exponent, work)?;
        let upper = scale_float(&bounds.upper, -exponent, work)?;
        let neighbor_probe = scale_float(&bounds.neighbor_probe, -exponent, work)?;
        let lower_matrix = shifted_matrix(&normalized, n, &lower, even, work)?;
        let lower_result = interval_symmetric_ldlt_inertia_mpfr(&lower_matrix, dimension, work)?;
        drop(lower_matrix);
        match lower_result {
            IntervalInertiaResult::Conclusive { negative: 0, .. } => {}
            IntervalInertiaResult::Conclusive { .. } => {
                bail!("claimed CCM eigenpair is not the algebraic ground state in the requested subspace")
            }
            IntervalInertiaResult::Inconclusive { .. } => continue,
        }
        let upper_matrix = shifted_matrix(&normalized, n, &upper, even, work)?;
        match interval_symmetric_ldlt_inertia_mpfr(&upper_matrix, dimension, work)? {
            IntervalInertiaResult::Conclusive { negative: 1, .. } => {
                if dimension == 1 {
                    return Ok(());
                }
                drop(upper_matrix);
                let neighbor_matrix = shifted_matrix(&normalized, n, &neighbor_probe, even, work)?;
                match interval_symmetric_ldlt_inertia_mpfr(&neighbor_matrix, dimension, work)? {
                    IntervalInertiaResult::Conclusive { negative: 1, .. } => return Ok(()),
                    IntervalInertiaResult::Conclusive { .. } => bail!(
                        "CCM stored precision resolution limit: selected-subspace gap is not separated from the matrix rounding scale at {p} bits"),
                    IntervalInertiaResult::Inconclusive { .. } => continue,
                }
            }
            IntervalInertiaResult::Conclusive { negative, .. } => {
                bail!(
                    "CCM ground state is unresolved or clustered at the requested precision: upper_count={negative}, eigenvalue={lambda}, residual={}",
                    bounds.record.residual_upper
                )
            }
            IntervalInertiaResult::Inconclusive { .. } => {}
        }
    }
    bail!("CCM ground-index inertia is inconclusive within the guard precision budget")
}

#[cfg(test)]
mod tests {
    use super::*;
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
        assert!(preflight(1001, 1000, CcmParityPolicy::EvenSector).is_err());
        preflight(301, 1725, CcmParityPolicy::EvenSector).unwrap();
        // Former 64-Mbit cap rejected this modest rational image even for zeros.
        let entries = vec![I::from_i64(0, 1789); 151 * 151];
        rational_matrix(&entries, 1789, 301 * 301 * 512 * 16).unwrap();
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
        assert!(rational_matrix(&intervals, p, 0).is_err());
    }
}
