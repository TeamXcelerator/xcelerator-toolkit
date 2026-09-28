//! Indexed uncertainty for the exact retained parity matrix, not its assembly.
use super::{
    compute_sector_eigenvalues, sector_transform_validation, CcmSectorEigenpairHp,
    CcmSectorEigenvalueRoute, SectorEigenvaluesHp, SectorTransformHp, SectorTridiagonalHp,
};
use anyhow::{bail, Result};
use rug::{float::Round, Float};
use xc_numerics::mpfr_interval::MpfrInterval as I;

pub(super) const ARITHMETIC: &str =
    "indexed_sturm_plus_polar_similarity_allowance_relative_8bit_v2";

pub(super) fn bounds(
    matrix: &[Float],
    tridiagonal: &SectorTridiagonalHp,
    transform: &SectorTransformHp,
    eigenvalues: &SectorEigenvaluesHp,
    count: usize,
    p: u32,
) -> Result<Vec<(Float, Float)>> {
    let n = tridiagonal.diagonal.len();
    let allowance = sector_transform_validation::bounds(
        matrix,
        &tridiagonal.diagonal,
        &tridiagonal.off_diagonal,
        &transform.basis,
        n,
        p,
    )
    .map_err(anyhow::Error::msg)?
    .eigenvalue_allowance;
    let selected;
    let enclosures = if eigenvalues.selected_enclosures.len() >= count {
        &eigenvalues.selected_enclosures
    } else {
        selected = compute_sector_eigenvalues(
            tridiagonal,
            n,
            count,
            CcmSectorEigenvalueRoute::Selected,
            p,
        )?;
        &selected.selected_enclosures
    };
    enclosures.iter().take(count).enumerate().map(|(index, e)| {
        if e.index != index || e.lower_count != index || e.upper_count != index + 1 {
            bail!("CCM sector resolution limit: Sturm enclosure does not isolate algebraic index {index}");
        }
        let lower = Float::with_val_round(p + 64, &e.lower - &allowance, Round::Down).0;
        let upper = Float::with_val_round(p + 64, &e.upper + &allowance, Round::Up).0;
        I::new(lower.clone(), upper.clone())?;
        Ok((lower, upper))
    }).collect()
}

pub(super) fn interval(pair: &CcmSectorEigenpairHp, p: u32) -> Result<I> {
    let lo = Float::with_val_round(p + 64, &pair.eigenvalue_lower, Round::Down).0;
    let hi = Float::with_val_round(p + 64, &pair.eigenvalue_upper, Round::Up).0;
    Ok(I::new(lo, hi)?)
}

/// Eight relative bits are a minimum admission floor, not a claim that every
/// displayed digit is accurate. Public endpoints preserve the actual bound.
pub(super) fn require_relative(pair: &CcmSectorEigenpairHp, p: u32) -> Result<I> {
    let enclosure = interval(pair, p)?;
    if enclosure.lower() <= &0 && enclosure.upper() >= &0 {
        bail!("CCM sector resolution limit: eigenvalue enclosure contains zero");
    }
    let work = p + 64;
    let point = I::from_float(&pair.eigenvalue, work)?;
    let error = super::super::retained_evidence::finite_math::abs(&enclosure.sub(&point))?;
    let magnitude = super::super::retained_evidence::finite_math::abs(&enclosure)?;
    let budget = magnitude.lower().clone() >> 8;
    if error.upper() > &budget {
        bail!("CCM sector resolution limit: fewer than eight relative bits are resolved; increase precision");
    }
    Ok(enclosure)
}
