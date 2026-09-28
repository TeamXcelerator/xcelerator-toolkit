//! Finite Fourier coefficient projection in the full-support L2(dx) metric.
use super::{point, scalar, ProjectionData, ProjectionOptions, RetainedReference, RetainedState};
use anyhow::{bail, Result};
use rug::{float::Round, Float};
use std::cmp::Ordering;

const PIVOT_METRIC: &str =
    "full_support_Gram_after_independent_binary_column_scaling; interval_positive_pivots; finite_matrix_only";
pub(super) fn pivot_metric() -> &'static str {
    PIVOT_METRIC
}
fn padded(v: &[Float], n: usize, p: u32) -> Vec<Float> {
    let offset = n - v.len() / 2;
    let mut out = vec![Float::with_val(p, 0); 2 * n + 1];
    for (i, x) in v.iter().enumerate() {
        out[offset + i] = Float::with_val(p, x);
    }
    out
}
// Point Gaussian elimination on a column-equilibrated Gram matrix. A small or
// nonpositive pivot is explicitly unresolved, not evidence of exact dependence.
pub(super) fn solve(gram: &[Float], rhs: &[Float], p: u32) -> Result<Option<(Vec<Float>, Float)>> {
    let n = rhs.len();
    if n == 0 {
        return Ok(Some((vec![], Float::with_val(p, 0))));
    }
    if gram.len() != n * n || gram.iter().chain(rhs).any(|x| !x.is_finite()) {
        bail!("invalid finite projection system");
    }
    let mut a = gram.to_vec();
    let mut b = rhs.to_vec();
    let maximum = a
        .iter()
        .fold(Float::with_val(p, 0), |m, x| m.max(&x.clone().abs()));
    if maximum == 0 {
        return Ok(None);
    }
    let threshold = Float::with_val(p, &maximum) >> (p - 32);
    if threshold == 0 {
        bail!("projection pivot threshold exceeds exponent range");
    }
    let mut minimum = Float::with_val(p, f64::INFINITY);
    for k in 0..n {
        let pivot = a[k * n + k].clone();
        if !pivot.is_finite() {
            bail!("nonfinite projection pivot");
        }
        if pivot <= threshold {
            return Ok(None);
        }
        minimum = minimum.min(&pivot);
        for i in k + 1..n {
            let factor = point::quotient(&a[i * n + k], &pivot, p)?;
            for j in k + 1..n {
                let mut value = -factor.clone();
                let dir = value.mul_add_round(&a[k * n + j], &a[i * n + j], Round::Nearest);
                if !value.is_finite() || (value.is_zero() && dir != Ordering::Equal) {
                    bail!("projection elimination exceeds MPFR range");
                }
                a[i * n + j] = value;
            }
            let mut value = -factor;
            let dir = value.mul_add_round(&b[k], &b[i], Round::Nearest);
            if !value.is_finite() || (value.is_zero() && dir != Ordering::Equal) {
                bail!("projection right-hand elimination exceeds MPFR range");
            }
            b[i] = value;
        }
    }
    for i in (0..n).rev() {
        let mut terms = vec![b[i].clone()];
        for j in i + 1..n {
            terms.push(-point::product(&[&a[i * n + j], &b[j]], 2 * p)?);
        }
        b[i] = point::quotient(&point::sum(&terms, p)?, &a[i * n + i], p)?;
    }
    Ok(Some((b, minimum)))
}
const UNCERTAINTY: &str = "finite_stored_inputs_with_arithmetic_enclosures; no_reference_approximation_or_construction_error_enclosure";
pub(super) fn uncertainty() -> &'static str {
    UNCERTAINTY
}
fn serialized(
    bound: &xc_numerics::mpfr_interval::MpfrInterval,
    p: u32,
) -> Result<(String, [String; 2])> {
    bound.validate()?;
    let middle = bound.midpoint_point();
    let mid = Float::with_val(p, middle.lower());
    let lo = Float::with_val_round(p, bound.lower(), Round::Down).0;
    let hi = Float::with_val_round(p, bound.upper(), Round::Up).0;
    if !mid.is_finite()
        || !lo.is_finite()
        || !hi.is_finite()
        || (mid.is_zero() && !middle.lower().is_zero())
    {
        bail!("projection report exceeds requested output range");
    }
    let digits = Some((u64::from(p) * 30103 / 100000 + 10) as usize);
    Ok((
        mid.to_string_radix_round(10, digits, Round::Nearest),
        [
            lo.to_string_radix_round(10, digits, Round::Down),
            hi.to_string_radix_round(10, digits, Round::Up),
        ],
    ))
}
pub(super) fn calculate(
    state: &RetainedState,
    reference: &RetainedReference,
    basis: &[RetainedReference],
    o: &ProjectionOptions,
) -> Result<ProjectionData> {
    let p = o.working_precision_bits;
    let n = std::iter::once(state.modes)
        .chain(std::iter::once(reference.spec.coefficients.len() / 2))
        .chain(basis.iter().map(|b| b.spec.coefficients.len() / 2))
        .max()
        .unwrap();
    let source = padded(&state.coefficients, n, p);
    let reference = padded(&reference.spec.values(p)?, n, p);
    let columns = basis
        .iter()
        .map(|b| Ok(padded(&b.spec.values(p)?, n, p)))
        .collect::<Result<Vec<_>>>()?;
    let result = super::projection_math::measure(
        &super::projection_math::Inputs {
            source: &source,
            reference: &reference,
            basis: &columns,
            cutoff: &state.cutoff,
            center_one: o.normalization == "center_one",
            fixed: o.fixed_second_component.as_deref(),
        },
        p,
    )?;
    let mut values = std::collections::BTreeMap::new();
    let mut bounds = std::collections::BTreeMap::new();
    for (name, value) in result.values {
        let (mid, bound) = serialized(&value, p)?;
        values.insert(name.clone(), mid);
        bounds.insert(name, bound);
    }
    let normalized = result.outcome != "normalization_unresolved";
    let solved = result.outcome == "point_measurement";
    let b = basis.len();
    let mut gram = Vec::with_capacity(b * b);
    let mut rhs = Vec::with_capacity(b);
    if normalized {
        for a in 0..b {
            rhs.push(values.remove(&format!("rhs_{a}")).unwrap());
            for k in 0..b {
                gram.push(values.remove(&format!("gram_{a}_{k}")).unwrap());
            }
        }
    }
    let coefficients = if solved {
        Some(
            (0..b)
                .map(|a| values.remove(&format!("a_{a}")).unwrap())
                .collect(),
        )
    } else {
        None
    };
    Ok(ProjectionData {
        outcome: result.outcome.into(),
        metric: "full_support_L2_dx; finite_Fourier_coefficients".into(),
        normalization: o.normalization.clone(),
        source_center: values.remove("source_center").unwrap(),
        reference_center: values.remove("reference_center").unwrap(),
        signed_unit_overlap: values.remove("signed_unit_overlap").unwrap(),
        difference_norm_squared: values.remove("difference_norm_squared"),
        gram,
        rhs,
        coefficients,
        fit_residual_norm_squared: values.remove("fit_residual_norm_squared"),
        minimum_pivot: values.remove("minimum_pivot"),
        pivot_metric: PIVOT_METRIC.into(),
        fixed_second_component: o.fixed_second_component.clone(),
        b_effective: values.remove("b_effective"),
        b2: values.remove("b2"),
        uncertainty: UNCERTAINTY.into(),
        arithmetic_precision_bits: result.arithmetic_precision,
        arithmetic_enclosures: bounds,
    })
}
pub(super) fn validate_enclosures(r: &ProjectionData, p: u32) -> Result<()> {
    if r.arithmetic_precision_bits <= p || r.arithmetic_precision_bits > p + 4096 {
        bail!("invalid projection arithmetic precision");
    }
    let mut expected = std::collections::BTreeMap::new();
    for (name, value) in [
        ("source_center", &r.source_center),
        ("reference_center", &r.reference_center),
        ("signed_unit_overlap", &r.signed_unit_overlap),
    ] {
        expected.insert(name.to_owned(), value);
    }
    for (name, value) in [
        (
            "difference_norm_squared",
            r.difference_norm_squared.as_ref(),
        ),
        (
            "fit_residual_norm_squared",
            r.fit_residual_norm_squared.as_ref(),
        ),
        ("minimum_pivot", r.minimum_pivot.as_ref()),
        ("b_effective", r.b_effective.as_ref()),
        ("b2", r.b2.as_ref()),
    ] {
        if let Some(v) = value {
            expected.insert(name.into(), v);
        }
    }
    let n = r.rhs.len();
    for (j, value) in r.gram.iter().enumerate() {
        if n == 0 {
            bail!("projection Gram shape has no columns");
        }
        expected.insert(format!("gram_{}_{}", j / n, j % n), value);
    }
    for (j, value) in r.rhs.iter().enumerate() {
        expected.insert(format!("rhs_{j}"), value);
    }
    if let Some(coefs) = &r.coefficients {
        for (j, value) in coefs.iter().enumerate() {
            expected.insert(format!("a_{j}"), value);
        }
    }
    if !expected.keys().eq(r.arithmetic_enclosures.keys()) {
        bail!("projection enclosure fields do not match measurements");
    }
    for (name, mid) in expected {
        let [lo, hi] = &r.arithmetic_enclosures[&name];
        let lo = scalar(lo, p)?;
        let hi = scalar(hi, p)?;
        let mid = scalar(mid, p)?;
        if lo > hi || mid < lo || mid > hi {
            bail!("invalid projection arithmetic enclosure");
        }
        if matches!(
            name.as_str(),
            "difference_norm_squared" | "fit_residual_norm_squared"
        ) && lo < 0
        {
            bail!("negative squared norm enclosure");
        }
    }
    if r.b_effective.is_some() {
        let [lo, hi] = r
            .arithmetic_enclosures
            .get("a_0")
            .ok_or_else(|| anyhow::anyhow!("ratio without coefficient enclosure"))?;
        if scalar(lo, p)? <= 0 && scalar(hi, p)? >= 0 {
            bail!("projection ratio denominator enclosure includes zero");
        }
    }
    Ok(())
}
