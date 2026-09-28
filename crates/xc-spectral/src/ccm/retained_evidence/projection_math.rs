//! Finite Fourier coefficient arithmetic with explicit output enclosures.
use super::finite_math as math;
use anyhow::{bail, Result};
use rug::{float::Round, Float};
use std::{cmp::Ordering, collections::BTreeMap};
use xc_numerics::mpfr_interval::MpfrInterval as I;

pub(super) struct Inputs<'a> {
    pub(super) source: &'a [Float],
    pub(super) reference: &'a [Float],
    pub(super) basis: &'a [Vec<Float>],
    pub(super) cutoff: &'a str,
    pub(super) center_one: bool,
    pub(super) fixed: Option<&'a str>,
}
#[derive(Debug)]
pub(super) struct Measurement {
    pub(super) values: BTreeMap<String, I>,
    pub(super) outcome: &'static str,
    pub(super) arithmetic_precision: u32,
}
fn dot(a: &[I], b: &[I], p: u32) -> I {
    a.iter()
        .zip(b)
        .fold(I::from_i64(0, p), |s, (a, b)| s.add(&a.mul(b)))
}
fn norm2(v: &[I], p: u32) -> I {
    v.iter().fold(I::from_i64(0, p), |s, x| s.add(&x.square()))
}
fn exact_product(a: &Float, b: &Float, p: u32) -> Result<Float> {
    let (x, round) = Float::with_val_round(2 * p, a * b, Round::Nearest);
    if !x.is_finite() || round != Ordering::Equal {
        bail!("finite projection exact product exceeds exponent range");
    }
    Ok(x)
}
fn determinant(a: &Float, b: &Float, c: &Float, d: &Float, p: u32) -> Result<I> {
    let terms = [exact_product(a, b, p)?, -exact_product(c, d, p)?];
    Ok(I::new(
        Float::with_val_round(p, Float::sum(terms.iter()), Round::Down).0,
        Float::with_val_round(p, Float::sum(terms.iter()), Round::Up).0,
    )?)
}
fn center(v: &[Float], p: u32) -> Result<(I, i32)> {
    let n = v.len() / 2;
    let terms = v
        .iter()
        .enumerate()
        .map(|(j, x)| {
            if j.abs_diff(n).is_multiple_of(2) {
                x.clone()
            } else {
                -x.clone()
            }
        })
        .collect::<Vec<_>>();
    let (rounded, dir) = Float::with_val_round(p, Float::sum(terms.iter()), Round::Nearest);
    let sign = if rounded < 0 || (rounded == 0 && dir == Ordering::Greater) {
        -1
    } else if rounded > 0 || (rounded == 0 && dir == Ordering::Less) {
        1
    } else if v
        .iter()
        .max_by(|a, b| {
            Float::with_val(p, *a)
                .abs()
                .total_cmp(&Float::with_val(p, *b).abs())
        })
        .unwrap()
        < &Float::with_val(p, 0)
    {
        -1
    } else {
        1
    };
    Ok((
        I::new(
            Float::with_val_round(p, Float::sum(terms.iter()), Round::Down).0,
            Float::with_val_round(p, Float::sum(terms.iter()), Round::Up).0,
        )?,
        sign,
    ))
}
fn scaled(v: &[Float], p: u32) -> Result<(Vec<Float>, i64)> {
    let e = i64::from(
        v.iter()
            .filter_map(Float::get_exp)
            .max()
            .ok_or_else(|| anyhow::anyhow!("zero finite projection vector"))?,
    );
    Ok((
        v.iter()
            .map(|x| math::scale_float(x, -e, p))
            .collect::<Result<_>>()?,
        e,
    ))
}
fn intervals(v: &[Float], p: u32) -> Result<Vec<I>> {
    v.iter().map(|x| Ok(I::from_float(x, p)?)).collect()
}
fn unit_difference(x: &[Float], y: &[Float], sx: i32, sy: i32, p: u32) -> Result<(Vec<I>, I)> {
    let a = x.iter().map(|x| x.clone() * sx).collect::<Vec<_>>();
    let b = y.iter().map(|x| x.clone() * sy).collect::<Vec<_>>();
    let ai = intervals(&a, p)?;
    let bi = intervals(&b, p)?;
    let ax = norm2(&ai, p).sqrt()?;
    let by = norm2(&bi, p).sqrt()?;
    let overlap = dot(&ai, &bi, p)
        .div(&ax.mul(&by))?
        .try_intersection(&I::new(Float::with_val(p, -1), Float::with_val(p, 1))?)?
        .ok_or_else(|| anyhow::anyhow!("unit overlap enclosure misses [-1,1]"))?;
    let k = b
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            Float::with_val(p, *a)
                .abs()
                .total_cmp(&Float::with_val(p, *b).abs())
        })
        .unwrap()
        .0;
    if (a[k] > 0 && b[k] > 0) || (a[k] < 0 && b[k] < 0) {
        let lambda = ai[k].div(&bi[k])?;
        let q = a
            .iter()
            .zip(&b)
            .map(|(x, y)| Ok(determinant(x, &b[k], y, &a[k], p)?.div(&bi[k])?))
            .collect::<Result<Vec<_>>>()?;
        if q.iter().all(|x| x.lower() == &0 && x.upper() == &0) {
            return Ok((q, I::from_i64(1, p)));
        }
        let correction = I::from_i64(2, p)
            .mul(&lambda)
            .mul(&dot(&bi, &q, p))
            .add(&norm2(&q, p))
            .div(&ax.mul(&by).mul(&lambda.mul(&by).add(&ax)))?;
        Ok((
            q.iter()
                .zip(&bi)
                .map(|(q, y)| Ok(q.div(&ax)?.sub(&y.mul(&correction))))
                .collect::<Result<_>>()?,
            overlap,
        ))
    } else {
        Ok((
            ai.iter()
                .zip(&bi)
                .map(|(x, y)| Ok(x.div(&ax)?.sub(&y.div(&by)?)))
                .collect::<Result<_>>()?,
            overlap,
        ))
    }
}
fn calculate(i: &Inputs<'_>, requested: u32, p: u32) -> Result<Option<Measurement>> {
    let (x, xe) = scaled(i.source, p)?;
    let (y, ye) = scaled(i.reference, p)?;
    let (cx, sx) = center(&x, p)?;
    let (cy, sy) = center(&y, p)?;
    let l = math::decimal(i.cutoff, p)?.ln()?;
    if !l.is_strictly_positive() {
        bail!("finite projection requires C>1");
    }
    let (mut difference, overlap) = unit_difference(&x, &y, sx, sy, p)?;
    let mut values = BTreeMap::new();
    values.insert("source_center".into(), math::scale(&cx, xe)?);
    values.insert("reference_center".into(), math::scale(&cy, ye)?);
    values.insert("signed_unit_overlap".into(), overlap);
    let mut r = Measurement {
        values,
        outcome: "normalization_unresolved",
        arithmetic_precision: p,
    };
    if i.center_one {
        if (cx.lower() == &0 && cx.upper() == &0) || (cy.lower() == &0 && cy.upper() == &0) {
            return Ok(Some(r));
        }
        // Exact centers make one O(n) determinant per output sufficient. Retry
        // within the explicit guard budget until both sums are exact.
        if cx.lower() != cx.upper() || cy.lower() != cy.upper() {
            return Ok(None);
        }
        let denominator = cx.mul(&cy);
        if denominator.contains_zero() {
            return Ok(None);
        }
        difference = x
            .iter()
            .zip(&y)
            .map(|(x, y)| Ok(determinant(x, cy.lower(), y, cx.lower(), p)?.div(&denominator)?))
            .collect::<Result<_>>()?;
    } else {
        let root = l.sqrt()?;
        difference = difference
            .iter()
            .map(|x| Ok(x.div(&root)?))
            .collect::<Result<_>>()?;
    }
    let (d, de) = math::normalized(&difference)?;
    let h2 = norm2(&d, p);
    if !math::narrow(std::slice::from_ref(&h2), h2.upper(), requested)? {
        return Ok(None);
    }
    let (weight, we) = math::normalized(std::slice::from_ref(&l))?;
    let weight = &weight[0];
    r.values.insert(
        "difference_norm_squared".into(),
        math::scale(&h2.mul(weight), 2 * de + we)?,
    );
    if i.basis.is_empty() {
        r.values.insert(
            "fit_residual_norm_squared".into(),
            r.values["difference_norm_squared"].clone(),
        );
        r.values.insert("minimum_pivot".into(), I::from_i64(0, p));
        r.outcome = "point_measurement";
        return Ok(Some(r));
    }
    let columns = i
        .basis
        .iter()
        .map(|v| math::normalized(&intervals(v, p)?))
        .collect::<Result<Vec<_>>>()?;
    let b = columns.len();
    let mut gram = Vec::with_capacity(b * b);
    let mut rhs = Vec::with_capacity(b);
    for (a, (col, ce)) in columns.iter().enumerate() {
        let value = dot(col, &d, p);
        r.values.insert(
            format!("rhs_{a}"),
            math::scale(&value.mul(weight), de + ce + we)?,
        );
        rhs.push(value);
        for (k, (other, ke)) in columns.iter().enumerate() {
            let value = dot(col, other, p);
            r.values.insert(
                format!("gram_{a}_{k}"),
                math::scale(&value.mul(weight), ce + ke + we)?,
            );
            gram.push(value);
        }
    }
    r.outcome = "rank_or_precision_unresolved";
    let Some((fit, pivot)) = math::solve(&gram, &rhs, p)? else {
        return Ok(Some(r));
    };
    let largest = fit
        .iter()
        .map(|x| math::abs(x).map(|x| x.upper().clone()))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .max_by(Float::total_cmp)
        .unwrap();
    if !math::narrow(&fit, &largest, requested)? {
        return Ok(Some(r));
    }
    let mut residual = Vec::with_capacity(d.len());
    for (j, value) in d.iter().enumerate() {
        let mut rem = value.clone();
        for (a, (col, _)) in fit.iter().zip(&columns) {
            rem = rem.sub(&a.mul(&col[j]));
        }
        residual.push(rem);
    }
    let residual2 = norm2(&residual, p);
    let residual_scale = if residual2.is_strictly_positive() {
        residual2.upper()
    } else {
        h2.upper()
    };
    if !math::narrow(std::slice::from_ref(&residual2), residual_scale, requested)? {
        return Ok(Some(r));
    }
    let coefficients = fit
        .iter()
        .zip(&columns)
        .map(|(x, (_, e))| math::scale(x, de - e))
        .collect::<Result<Vec<_>>>()?;
    if let Some(fixed) = i.fixed {
        let fixed = math::decimal(fixed, p)?;
        r.values.insert(
            "b2".into(),
            coefficients[1].sub(&fixed.mul(&coefficients[0])),
        );
        if !coefficients[0].contains_zero() {
            r.values
                .insert("b_effective".into(), coefficients[1].div(&coefficients[0])?);
        }
    }
    for (a, x) in coefficients.into_iter().enumerate() {
        r.values.insert(format!("a_{a}"), x);
    }
    r.values.insert("minimum_pivot".into(), pivot.mul(&l));
    r.values.insert(
        "fit_residual_norm_squared".into(),
        math::scale(&residual2.mul(weight), 2 * de + we)?,
    );
    r.outcome = "point_measurement";
    Ok(Some(r))
}
pub(super) fn measure(i: &Inputs<'_>, requested: u32) -> Result<Measurement> {
    if !(64..=1_000_000).contains(&requested)
        || i.source.is_empty()
        || i.source.len().is_multiple_of(2)
        || i.source.len() > 16385
        || i.reference.len() != i.source.len()
        || i.basis.len() > 8
        || i.basis.iter().any(|v| v.len() != i.source.len())
        || (i.fixed.is_some() && i.basis.len() != 2)
        || i.source
            .iter()
            .chain(i.reference)
            .chain(i.basis.iter().flatten())
            .any(|x| !x.is_finite() || x.prec() > requested)
    {
        bail!("invalid finite projection shape, precision or domain");
    }
    if let Some(fixed) = i.fixed {
        math::decimal(fixed, requested)?.validate()?;
    }
    let mut last = None;
    for guard in [64u32, 128, 256, 512, 1024, 2048, 4096] {
        if let Some(r) = calculate(i, requested, requested + guard)? {
            for v in r.values.values() {
                v.validate()?;
            }
            if r.outcome != "rank_or_precision_unresolved" {
                return Ok(r);
            }
            last = Some(r);
        }
    }
    last.ok_or_else(|| {
        anyhow::anyhow!("finite projection arithmetic unresolved within 4096 guard bits")
    })
}
