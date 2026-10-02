//! Indexed full-spectrum bounds and resolved first-positive selection.
use super::{
    sector_gap_math, sector_transform_validation, stored_resolution, StoredEigenvalueResolution,
};
use anyhow::{bail, Result};
use rug::{float::Round, Float};
use xc_numerics::mpfr_interval::MpfrInterval as I;

pub const WEIL_SPECTRUM_SEMANTICS: &str = "stored_weil_full_spectrum_sturm_polar_bounds_v1";
pub const PLUNGE_SEMANTICS: &str =
    "full_first_positive_resolved_negative_floor_inertia_indexed_residual_gap_plunge_v2";

/// Bounds on one algebraic index of the exact stored Weil matrix.
#[derive(Clone, Debug)]
pub struct CcmWeilEigenvalueBoundHp {
    pub algebraic_index: usize,
    pub lower: Float,
    pub upper: Float,
    pub resolution: StoredEigenvalueResolution,
}

/// Full stored-matrix spectrum with directed source and precision evidence.
/// Finite-form assembly error remains a separate quantity.
#[derive(Clone, Debug)]
pub struct CcmWeilSpectrumHp {
    pub eigenvalues: Vec<Float>,
    pub eigenvalue_bounds: Vec<CcmWeilEigenvalueBoundHp>,
    pub matrix_rounding_scale_upper: Float,
    pub precision_bits: u32,
    pub analysis_precision_bits: u32,
    pub algorithm_semantics: String,
}

pub(super) fn full_spectrum(a: &[Float], n: usize, p: u32) -> Result<CcmWeilSpectrumHp> {
    lowest_spectrum(a, n, p, n)
}

/// Directed enclosures of the `count` algebraically smallest eigenvalues of an
/// exact stored symmetric matrix, including Householder source allowance.
pub(super) fn lowest_spectrum(
    a: &[Float],
    n: usize,
    p: u32,
    count: usize,
) -> Result<CcmWeilSpectrumHp> {
    if count == 0 || count > n {
        bail!("requested eigenvalue count must lie in 1..=dimension");
    }
    let floor = stored_resolution::matrix_rounding_scale(a, n, p)?;
    let work = p.saturating_add(64).min(1_000_000);
    let (d, e, q) = xc_numerics::eigen::householder_tridiag_hp_stable(a, n, work)?;
    let source_allowance = sector_transform_validation::bounds(a, &d, &e, &q, n, work)
        .map_err(anyhow::Error::msg)?
        .eigenvalue_allowance;
    let (sd, se, exponent) = sector_gap_math::scaled_tridiagonal(&d, &e, work)?;
    let tolerance = super::selected_sector_tolerance(
        &super::SectorTridiagonalHp {
            diagonal: sd.clone(),
            off_diagonal: se.clone(),
        },
        work,
    )?;
    let selected = xc_numerics::eigen::tridiag_selected_eigenvalues_hp(
        &sd,
        &se,
        0,
        count - 1,
        &tolerance,
        (work as usize).saturating_mul(2).saturating_add(256),
        work,
    )?;
    let mut values = Vec::with_capacity(n);
    let mut bounds = Vec::with_capacity(n);
    for (index, bound) in selected.enclosures.into_iter().enumerate() {
        if bound.index != index || bound.lower_count > index || bound.upper_count <= index {
            bail!("full Weil spectrum lost its algebraic index enclosure");
        }
        let lo =
            crate::ccm::retained_evidence::finite_math::scale_float(&bound.lower, exponent, work)?;
        let hi =
            crate::ccm::retained_evidence::finite_math::scale_float(&bound.upper, exponent, work)?;
        let lower = Float::with_val_round(work, &lo - &source_allowance, Round::Down).0;
        let upper = Float::with_val_round(work, &hi + &source_allowance, Round::Up).0;
        let interval = I::new(lower.clone(), upper.clone())?;
        let resolution = if interval.contains_zero() {
            StoredEigenvalueResolution::StoredSignUnresolved
        } else if crate::ccm::retained_evidence::finite_math::abs(&interval)?.lower() <= &floor {
            StoredEigenvalueResolution::BelowStorageFloor
        } else {
            StoredEigenvalueResolution::Resolved
        };
        values.push(Float::with_val(p, interval.midpoint_point().lower()));
        bounds.push(CcmWeilEigenvalueBoundHp {
            algebraic_index: index,
            lower,
            upper,
            resolution,
        });
    }
    Ok(CcmWeilSpectrumHp {
        eigenvalues: values,
        eigenvalue_bounds: bounds,
        matrix_rounding_scale_upper: floor,
        precision_bits: p,
        analysis_precision_bits: work,
        algorithm_semantics: format!(
            "{WEIL_SPECTRUM_SEMANTICS}+{}+{}",
            xc_numerics::eigen::STABLE_HOUSEHOLDER_SEMANTICS,
            super::sector_resolution::ARITHMETIC
        ),
    })
}

pub(super) fn first_positive(
    a: &[Float],
    n: usize,
    p: u32,
    steps: usize,
) -> Result<xc_numerics::eigen::HpEigenvectorRecovery> {
    // A stored eigenvalue within the matrix rounding scale of zero cannot be
    // classified as negative: skipping it would report a later eigenvalue as
    // the first positive one. Require every negative stored eigenvalue to lie
    // below -floor, proved by a second directed count.
    let floor = stored_resolution::matrix_rounding_scale(a, n, p)?;
    let mut negative_floor = floor.clone();
    negative_floor = -negative_floor;
    let mut selected = None;
    for guard in [64u32, 256, 1024] {
        let work = p.saturating_add(guard).min(1_000_000);
        let count_below = |shift: &Float| {
            xc_numerics::symmetric_inertia::point_matrix_inertia_at(a, n, shift, work, 8u64 << 30)
        };
        let at_zero = count_below(&Float::with_val(work, 0))?;
        let at_negative_floor = count_below(&Float::with_val(work, &negative_floor))?;
        use xc_numerics::symmetric_inertia::MpfrInertiaResult::Conclusive;
        if let (
            Conclusive { negative, .. },
            Conclusive {
                negative: below_floor,
                ..
            },
        ) = (at_zero, at_negative_floor)
        {
            selected = Some((negative, below_floor));
            break;
        }
    }
    let (index, below_floor) = selected.ok_or_else(|| {
        anyhow::anyhow!("CCM plunge resolution limit: zero-boundary inertia is unresolved")
    })?;
    if below_floor != index {
        bail!("CCM plunge resolution limit: {} stored eigenvalue(s) lie within the matrix rounding scale below zero, so the first positive index is not resolved; increase precision or inspect the bounded spectrum", index - below_floor);
    }
    if index >= n {
        bail!("no positive stored Weil eigenvalue");
    }
    let pair = xc_numerics::eigen::dense_symmetric_eigenpair_at_index_hp(a, n, index, p, steps)?;
    let lower = Float::with_val_round(
        p + 64,
        &pair.eigenvalue - &pair.residual_upper_bound,
        Round::Down,
    )
    .0;
    if lower <= floor {
        bail!("CCM plunge resolution limit: selected positive eigenvalue is not separated from the matrix rounding scale; increase precision or inspect the bounded spectrum");
    }
    Ok(pair)
}

#[cfg(test)]
mod plunge_resolution_tests {
    use super::*;

    fn diagonal(values: &[Float], p: u32) -> Vec<Float> {
        let n = values.len();
        let mut a = vec![Float::with_val(p, 0); n * n];
        for (i, v) in values.iter().enumerate() {
            a[i * n + i] = Float::with_val(p, v);
        }
        a
    }

    #[test]
    fn sub_floor_negative_eigenvalue_is_not_skipped() {
        for p in [128u32, 256] {
            // Spectrum is exact by construction: {-2^-(p+4), 1, 2}. The
            // negative value lies inside the rounding scale 3*2*2^-p, so
            // index 1 is not the resolved first positive eigenvalue.
            let tiny = -(Float::with_val(p, 1) >> (p + 4));
            let a = diagonal(&[tiny, Float::with_val(p, 1), Float::with_val(p, 2)], p);
            let error = first_positive(&a, 3, p, 200).unwrap_err().to_string();
            assert!(
                error.contains("within the matrix rounding scale"),
                "{error}"
            );
            // A resolved negative eigenvalue keeps the stored-matrix definition.
            let a = diagonal(
                &[
                    Float::with_val(p, -1),
                    Float::with_val(p, 1),
                    Float::with_val(p, 2),
                ],
                p,
            );
            let pair = first_positive(&a, 3, p, 200).unwrap();
            assert_eq!(pair.index, 1);
            assert_eq!(pair.eigenvalue, 1);
        }
    }
}
