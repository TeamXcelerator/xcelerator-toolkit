//! Scale-invariant finite retained matrix diagnostics; no ground-state claim.
use super::{point, scalar, EnergyData, RetainedMatrix, RetainedState};
use anyhow::{bail, Result};
use rug::{float::Round, Float};
use std::cmp::Ordering;
use xc_numerics::prefix::lossless_decimal as dec;

pub(super) fn calculate(
    s: &RetainedState,
    m: &RetainedMatrix<'_>,
    requested: u32,
) -> Result<EnergyData> {
    let p = requested + 64;
    let n = s.coefficients.len();
    // Serialized source values denote the original source-precision points.
    // Re-reading their round-trip decimals at higher precision invents digits
    // and can create a residual even for an exact scalar-matrix eigenpair.
    let eigen = Float::with_val(p, scalar(&s.eigenvalue, s.precision)?);
    let vector_exp = i64::from(
        s.coefficients
            .iter()
            .filter_map(Float::get_exp)
            .max()
            .ok_or_else(|| anyhow::anyhow!("zero retained energy state"))?,
    );
    let matrix_exp = i64::from(
        m.entries
            .iter()
            .chain(std::iter::once(&eigen))
            .filter_map(Float::get_exp)
            .max()
            .unwrap_or(0),
    );
    let v = s
        .coefficients
        .iter()
        .map(|x| point::scale(x, -vector_exp, p))
        .collect::<Result<Vec<_>>>()?;
    let lambda = point::scale(&eigen, -matrix_exp, p)?;
    let norm2 = point::dot(&v, &v, p)?;
    if norm2 <= 0 {
        bail!("retained energy state norm is unresolved");
    }
    // Every three-factor product lies on a binary grid whose least exponent
    // is at least min(A_exp)+2*min(x_exp)-3p. At most n^2 terms are added.
    // This gives an exact accumulator precision without retaining a second
    // dense matrix or n^2 enlarged-precision products.
    let x_min = v.iter().filter_map(Float::get_exp).min().unwrap() as i64;
    let x_max = v.iter().filter_map(Float::get_exp).max().unwrap() as i64;
    let a_min = m
        .entries
        .iter()
        .filter_map(Float::get_exp)
        .min()
        .map(i64::from)
        .unwrap_or(matrix_exp)
        - matrix_exp;
    let a_max = m
        .entries
        .iter()
        .filter_map(Float::get_exp)
        .max()
        .map(i64::from)
        .unwrap_or(matrix_exp)
        - matrix_exp;
    let count_bits = usize::BITS - m.entries.len().saturating_sub(1).leading_zeros();
    let exact_bits =
        u32::try_from(a_max - a_min + 2 * (x_max - x_min) + i64::from(3 * p + count_bits + 2))?;
    if exact_bits > 3 * p + 1_000_000 {
        bail!("retained exact energy accumulator exceeds the exponent-span budget");
    }
    let mut energy_exact = Float::with_val(exact_bits, 0);
    let mut absolute_exact = Float::with_val(exact_bits, 0);
    let mut residual_norm = Float::with_val(p, 0);
    for (i, row) in m.entries.chunks_exact(n).enumerate() {
        let mut residual_terms = Vec::with_capacity(n + 1);
        for (entry, x) in row.iter().zip(&v) {
            let scaled = point::scale(entry, -matrix_exp, p)?;
            let action_term = point::product(&[&scaled, x], 2 * p)?;
            let energy_term = point::product(&[&action_term, &v[i]], 3 * p)?;
            for (acc, term) in [
                (&mut energy_exact, energy_term.clone()),
                (&mut absolute_exact, energy_term.abs()),
            ] {
                let (value, dir) = Float::with_val_round(exact_bits, &*acc + &term, Round::Nearest);
                if !value.is_finite() || dir != Ordering::Equal {
                    bail!("retained exact energy accumulation is out of range");
                }
                *acc = value;
            }
            residual_terms.push(action_term);
        }
        residual_terms.push(-point::product(&[&lambda, &v[i]], 2 * p)?);
        residual_norm.hypot_mut(&point::sum(&residual_terms, p)?);
    }
    let energy = point::output(&energy_exact, p)?;
    let absolute = point::output(&absolute_exact, p)?;
    let rayleigh = point::quotient(&energy, &norm2, p)?;
    let defect = point::sum(&[rayleigh.clone(), -lambda.clone()], p)?;
    let normalized_residual = point::quotient(&residual_norm, &point::norm(&v, p)?, p)?;
    let relative = if eigen.is_zero() {
        // The historical zero-eigenvalue convention uses scale=1, so this is
        // the absolute residual divided by the coefficient norm.
        point::scale(&normalized_residual, matrix_exp, p)?
    } else {
        point::quotient(&normalized_residual, &lambda.clone().abs(), p)?
    };
    let cancellation = if !energy.is_zero() && absolute > 0 {
        let c = absolute.clone().log10() - energy.abs().log10();
        Some(dec(&point::output(
            &c.max(&Float::with_val(p, 0)),
            requested,
        )?))
    } else {
        None
    };
    let physical = |x: &Float, e: i64| -> Result<String> {
        Ok(dec(&point::output(&point::scale(x, e, p)?, requested)?))
    };
    Ok(EnergyData {
        lambda_squared: s.cutoff.clone(),
        n_modes: s.modes,
        precision_bits: requested,
        source_eigenvalue: s.eigenvalue.clone(),
        coefficient_norm_squared: physical(&norm2, 2 * vector_exp)?,
        rayleigh_quotient: physical(&rayleigh, matrix_exp)?,
        eigenvalue_defect: physical(&defect, matrix_exp)?,
        relative_residual: dec(&point::output(&relative, requested)?),
        residual_normalization: residual_normalization(&s.eigenvalue, s.precision)?.into(),
        sum_absolute_energy_terms: physical(&absolute, matrix_exp + 2 * vector_exp)?,
        cancellation_digits: cancellation,
        component_decomposition: "not_captured; total_Tau_only; no_channel_sign_inference".into(),
        ground_selection: "not_established_by_this_diagnostic".into(),
    })
}

pub(super) fn residual_normalization(eigenvalue: &str, p: u32) -> Result<&'static str> {
    Ok(if super::scalar(eigenvalue, p)?.is_zero() {
        "absolute_residual_per_coefficient_l2_norm"
    } else {
        "relative_to_abs_eigenvalue_and_coefficient_l2_norm"
    })
}
