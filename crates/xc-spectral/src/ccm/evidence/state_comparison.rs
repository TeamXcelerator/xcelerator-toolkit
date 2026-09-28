//! Guarded finite-state comparison with verified positive-definite form input.
use anyhow::{bail, Result};
use rug::{float::Round, Float};
use std::cmp::Ordering;
use xc_numerics::mpfr_interval::MpfrInterval as I;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComparisonTruncationKind {
    ValueSpace,
    CoefficientSpace,
    FormNorm,
}

/// Caller-supplied truncation premise. A source label is descriptive; this type
/// does not authenticate the estimate or add it to the reported point values.
#[derive(Clone, Debug)]
pub struct ActiveTruncationBound {
    pub kind: ComparisonTruncationKind,
    pub upper_bound: Float,
    pub source: String,
}

/// Discrete point comparison of independently normalized real vectors.
/// Value samples use unweighted Euclidean norms. Supplied truncation estimates
/// are retained as external premises, not verified continuum error bounds.
#[derive(Clone, Debug)]
pub struct ProlateWeilStateComparisonHp {
    pub value_space_overlap: Float,
    pub value_space_residual: Float,
    pub coefficient_overlap: Float,
    pub coefficient_residual: Float,
    pub prolate_rayleigh_quotient: Float,
    pub weil_rayleigh_quotient: Float,
    pub prolate_eigen_residual: Float,
    pub weil_eigen_residual: Float,
    pub form_norm_difference: Float,
    pub truncation_bounds: Vec<ActiveTruncationBound>,
}

fn scale(v: &Float, exponent: i64, p: u32) -> Result<Float> {
    let mut out = Float::with_val(p, v);
    if v.is_zero() {
        return Ok(out);
    }
    let amount = u32::try_from(exponent.unsigned_abs())?;
    if exponent >= 0 {
        out <<= amount;
    } else {
        out >>= amount;
    }
    if !out.is_finite() || out.is_zero() {
        bail!("state-comparison scale exceeds MPFR range");
    }
    let mut reverse = out.clone();
    if exponent >= 0 {
        reverse >>= amount;
    } else {
        reverse <<= amount;
    }
    if reverse != *v {
        bail!("state-comparison binary scale loses a component");
    }
    Ok(out)
}
fn norm(v: &[Float], p: u32) -> Result<Float> {
    let mut n = Float::with_val(p, 0);
    for x in v {
        n.hypot_mut(x);
    }
    if !n.is_finite() {
        bail!("state-comparison norm exceeds MPFR range");
    }
    Ok(n)
}
fn normalize(v: &[Float], p: u32) -> Result<Vec<Float>> {
    let e = v
        .iter()
        .filter_map(Float::get_exp)
        .max()
        .ok_or_else(|| anyhow::anyhow!("zero comparison vector"))?;
    let scaled = v
        .iter()
        .map(|x| scale(x, 1 - i64::from(e), p))
        .collect::<Result<Vec<_>>>()?;
    let n = norm(&scaled, p)?;
    scaled
        .iter()
        .map(|x| {
            let out = Float::with_val(p, x / &n);
            if !out.is_finite() || (!x.is_zero() && out.is_zero()) {
                bail!("comparison normalization loses a component");
            }
            Ok(out)
        })
        .collect()
}
fn product(a: &Float, b: &Float, p: u32) -> Result<Float> {
    let (v, dir) = Float::with_val_round(p, a * b, Round::Nearest);
    if !v.is_finite() || dir != Ordering::Equal {
        bail!("comparison product exceeds exact intermediate range");
    }
    Ok(v)
}
fn sum(v: &[Float], p: u32) -> Result<Float> {
    let (out, dir) = Float::with_val_round(p, Float::sum(v.iter()), Round::Nearest);
    if !out.is_finite() || (out.is_zero() && dir != Ordering::Equal) {
        bail!("comparison sum exceeds MPFR range");
    }
    Ok(out)
}
fn dot(a: &[Float], b: &[Float], p: u32) -> Result<Float> {
    let terms = a
        .iter()
        .zip(b)
        .map(|(x, y)| product(x, y, 2 * p))
        .collect::<Result<Vec<_>>>()?;
    sum(&terms, p)
}
fn aligned(a: &[Float], b: &[Float], p: u32) -> Result<(Float, Vec<Float>)> {
    let overlap = dot(a, b, p)?;
    let sign = if overlap < 0 { -1 } else { 1 };
    let d = a
        .iter()
        .zip(b)
        .map(|(x, y)| {
            let mut value = Float::with_val(p, y);
            value *= sign;
            let (out, dir) = Float::with_val_round(p, &value - x, Round::Nearest);
            if !out.is_finite() || (out.is_zero() && dir != Ordering::Equal) {
                bail!("comparison difference exceeds MPFR range");
            }
            Ok(out)
        })
        .collect::<Result<_>>()?;
    Ok((overlap.abs(), d))
}

// Interval LDL proves positive definiteness of the exact stored symmetric
// matrix. An unresolved pivot returns an error; no point pivot is promoted to
// a proof. The point report below is separate from this admission check.
fn positive_definite(matrix: &[Float], n: usize, p: u32) -> Result<()> {
    let mut l = vec![I::from_i64(0, p); matrix.len()];
    let mut diag = Vec::<I>::with_capacity(n);
    for j in 0..n {
        let mut pivot = I::from_float(&matrix[j * n + j], p)?;
        for k in 0..j {
            pivot = pivot.sub(&l[j * n + k].square().mul(&diag[k]));
        }
        pivot.validate()?;
        if !pivot.is_strictly_positive() {
            bail!("positive-definite comparison form is not established at pivot {j}");
        }
        for i in j + 1..n {
            let mut value = I::from_float(&matrix[i * n + j], p)?;
            for k in 0..j {
                value = value.sub(&l[i * n + k].mul(&l[j * n + k]).mul(&diag[k]));
            }
            value.validate()?;
            l[i * n + j] = value.div(&pivot)?;
        }
        diag.push(pivot);
    }
    Ok(())
}
fn matvec(a: &[Float], v: &[Float], p: u32) -> Result<Vec<Float>> {
    a.chunks_exact(v.len()).map(|row| dot(row, v, p)).collect()
}
fn eigen_measurement(a: &[Float], v: &[Float], p: u32) -> Result<(Float, Float)> {
    let action = matvec(a, v, p)?;
    let q = Float::with_val(p, dot(v, &action, p)? / dot(v, v, p)?);
    if !q.is_finite() || q <= 0 {
        bail!("positive comparison Rayleigh quotient is unresolved");
    }
    let residual = v
        .iter()
        .zip(action)
        .map(|(x, y)| {
            let mut r = -q.clone();
            let dir = r.mul_add_round(x, &y, Round::Nearest);
            if !r.is_finite() || (r.is_zero() && dir != Ordering::Equal) {
                bail!("comparison eigen-residual exceeds MPFR range");
            }
            Ok(r)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((q, Float::with_val(p, norm(&residual, p)? / norm(v, p)?)))
}
fn form_norm(a: &[Float], d: &[Float], p: u32) -> Result<Float> {
    let Some(e) = d.iter().filter_map(Float::get_exp).max() else {
        return Ok(Float::with_val(p, 0));
    };
    let exponent = i64::from(e) - 1;
    let v = d
        .iter()
        .map(|x| scale(x, -exponent, p))
        .collect::<Result<Vec<_>>>()?;
    let n = v.len();
    let mut terms = Vec::with_capacity(a.len());
    for i in 0..n {
        for j in 0..=i {
            let first = product(&a[i * n + j], &v[i], 2 * p)?;
            let term = product(&first, &v[j], 3 * p)?;
            terms.push(if i == j {
                term
            } else {
                scale(&term, 1, 3 * p)?
            });
        }
    }
    let q = sum(&terms, p)?;
    if q <= 0 {
        bail!("positive form norm for a nonzero difference is numerically unresolved");
    }
    scale(&q.sqrt(), exponent, p)
}
fn output(v: &Float, p: u32) -> Result<Float> {
    let out = Float::with_val(p, v);
    if !out.is_finite() || (!v.is_zero() && out.is_zero()) {
        bail!("comparison output exceeds requested range");
    }
    Ok(out)
}

/// Compare two value-sample vectors and two coefficient vectors. Each pair
/// needs equal nonzero dimension, and the supplied coefficient form must be
/// finite, exactly symmetric and provably positive definite at working precision.
/// This admission check uses interval LDL (cubic cost); unresolved inputs fail.
///
/// Precision must be 64..=1,000,000 bits; finite source values may use up to
/// 1,000,000 bits. Working precision is the maximum plus 64 guard bits. All
/// returned scalars are point measurements at requested precision. There is no
/// composite forward-error or continuum certificate. Each truncation-kind label
/// occurs exactly once and remains a caller-supplied scientific premise.
pub fn compare_prolate_weil_states_hp(
    weil_values: &[Float],
    prolate_values: &[Float],
    weil_coefficients: &[Float],
    prolate_coefficients: &[Float],
    weil_form: &[Float],
    truncation_bounds: Vec<ActiveTruncationBound>,
    precision_bits: u32,
) -> Result<ProlateWeilStateComparisonHp> {
    if !(64..=1_000_000).contains(&precision_bits) {
        bail!("unsupported comparison precision");
    }
    let n = weil_coefficients.len();
    if n == 0
        || weil_values.is_empty()
        || weil_values.len() != prolate_values.len()
        || n != prolate_coefficients.len()
        || n.checked_mul(n) != Some(weil_form.len())
    {
        bail!("comparison vectors and square form have inconsistent shapes");
    }
    let all = || {
        weil_values
            .iter()
            .chain(prolate_values)
            .chain(weil_coefficients)
            .chain(prolate_coefficients)
            .chain(weil_form)
    };
    if all().any(|x| !x.is_finite() || x.prec() > 1_000_000) {
        bail!("comparison requires finite supported-precision inputs");
    }
    let p = all().map(Float::prec).fold(precision_bits, u32::max) + 64;
    for i in 0..n {
        for j in 0..i {
            if weil_form[i * n + j] != weil_form[j * n + i] {
                bail!("comparison form must be exactly symmetric");
            }
        }
    }
    for kind in [
        ComparisonTruncationKind::ValueSpace,
        ComparisonTruncationKind::CoefficientSpace,
        ComparisonTruncationKind::FormNorm,
    ] {
        if truncation_bounds.iter().filter(|b| b.kind == kind).count() != 1 {
            bail!("each comparison space requires exactly one supplied truncation bound");
        }
    }
    if truncation_bounds
        .iter()
        .any(|b| !b.upper_bound.is_finite() || b.upper_bound < 0 || b.source.trim().is_empty())
    {
        bail!("truncation premises require finite nonnegative bounds and source descriptions");
    }
    let exponent = weil_form
        .iter()
        .filter_map(Float::get_exp)
        .max()
        .ok_or_else(|| anyhow::anyhow!("zero comparison form"))?;
    let matrix_exp = (i64::from(exponent) - 1).div_euclid(2) * 2;
    let a = weil_form
        .iter()
        .map(|x| scale(x, -matrix_exp, p))
        .collect::<Result<Vec<_>>>()?;
    positive_definite(&a, n, p)?;
    let wv = normalize(weil_values, p)?;
    let pv = normalize(prolate_values, p)?;
    let (value_space_overlap, vd) = aligned(&wv, &pv, p)?;
    let wc = normalize(weil_coefficients, p)?;
    let pc = normalize(prolate_coefficients, p)?;
    let (coefficient_overlap, cd) = aligned(&wc, &pc, p)?;
    let (wq, wr) = eigen_measurement(&a, &wc, p)?;
    let (pq, pr) = eigen_measurement(&a, &pc, p)?;
    Ok(ProlateWeilStateComparisonHp {
        value_space_overlap: output(&value_space_overlap, precision_bits)?,
        value_space_residual: output(&norm(&vd, p)?, precision_bits)?,
        coefficient_overlap: output(&coefficient_overlap, precision_bits)?,
        coefficient_residual: output(&norm(&cd, p)?, precision_bits)?,
        prolate_rayleigh_quotient: output(&scale(&pq, matrix_exp, p)?, precision_bits)?,
        weil_rayleigh_quotient: output(&scale(&wq, matrix_exp, p)?, precision_bits)?,
        prolate_eigen_residual: output(&scale(&pr, matrix_exp, p)?, precision_bits)?,
        weil_eigen_residual: output(&scale(&wr, matrix_exp, p)?, precision_bits)?,
        form_norm_difference: output(
            &scale(&form_norm(&a, &cd, p)?, matrix_exp / 2, p)?,
            precision_bits,
        )?,
        truncation_bounds,
    })
}
