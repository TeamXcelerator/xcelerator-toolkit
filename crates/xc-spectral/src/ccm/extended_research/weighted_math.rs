//! Directed arithmetic for a finite weighted grid. No continuum error bound.
use crate::ccm::retained_evidence::finite_math::{
    abs, decimal, narrow, normalized, scale, scale_float, solve,
};
use anyhow::{bail, Result};
use rug::{float::Round, Float};
use std::collections::BTreeMap;
use xc_numerics::mpfr_interval::MpfrInterval as I;

#[derive(Debug)]
pub(super) struct Measurement {
    pub(super) values: BTreeMap<String, I>,
    pub(super) fit_resolved: bool,
    pub(super) arithmetic_precision: u32,
}
fn calculate(
    v: &[Float],
    cutoff: &str,
    target: &[Float],
    basis: &[Vec<Float>],
    fixed: Option<&Float>,
    requested: u32,
    p: u32,
) -> Result<Option<Measurement>> {
    let z = I::from_i64(0, p);
    let one = I::from_i64(1, p);
    let n = target.len() - 1;
    let modes = v.len() / 2;
    let source_exp = i64::from(v.iter().filter_map(Float::get_exp).max().unwrap());
    let source = v
        .iter()
        .map(|x| scale_float(x, -source_exp, p))
        .collect::<Result<Vec<_>>>()?;
    let terms = source
        .iter()
        .enumerate()
        .map(|(j, x)| {
            if j.abs_diff(modes).is_multiple_of(2) {
                x.clone()
            } else {
                -x.clone()
            }
        })
        .collect::<Vec<_>>();
    let c = I::new(
        Float::with_val_round(p, Float::sum(terms.iter()), Round::Down).0,
        Float::with_val_round(p, Float::sum(terms.iter()), Round::Up).0,
    )?;
    if c.lower() == &0 && c.upper() == &0 {
        bail!("weighted profile center is exactly zero");
    }
    if c.contains_zero() {
        return Ok(None);
    }
    let l = decimal(cutoff, p)?.ln()?;
    if !l.is_strictly_positive() {
        bail!("weighted profile requires C>1");
    }
    let h = l.div(&I::from_u64((2 * n) as u64, p))?;
    let mut cosine = Vec::with_capacity(n + 1);
    for j in 0..=n {
        cosine.push(if j == 0 {
            one.clone()
        } else if j == n {
            one.neg()
        } else if 2 * j == n {
            z.clone()
        } else {
            I::pi(p)
                .mul(&I::from_u64(j as u64, p))
                .div(&I::from_u64(n as u64, p))?
                .cos()
        });
    }
    let mut paired = Vec::with_capacity(modes + 1);
    paired.push(I::from_float(&source[modes], p)?);
    for k in 1..=modes {
        let pair =
            I::from_float(&source[modes + k], p)?.add(&I::from_float(&source[modes - k], p)?);
        paired.push(if k.is_multiple_of(2) {
            pair
        } else {
            pair.neg()
        });
    }
    let mut differences = Vec::with_capacity(n + 1);
    let mut weights = Vec::with_capacity(n + 1);
    for (j, t) in target.iter().enumerate() {
        let t = I::from_float(t, p)?;
        let mut numerator = paired[0].mul(&one.sub(&t));
        for (k, a) in paired.iter().enumerate().skip(1) {
            let phase = (k * j) % (2 * n);
            let phase = phase.min(2 * n - phase);
            numerator = numerator.add(&a.mul(&cosine[phase].sub(&t)));
        }
        differences.push(numerator.div(&c)?);
        let w = h.mul(
            &h.mul(&I::from_u64(j as u64, p))
                .div(&I::from_i64(2, p))?
                .exp(),
        );
        weights.push(if j == 0 || j == n {
            w.div(&I::from_i64(2, p))?
        } else {
            w
        });
    }
    let (d, de) = normalized(&differences)?;
    let (w, we) = normalized(&weights)?;
    if w.iter().any(|x| !x.is_strictly_positive()) {
        bail!("weighted grid weight is unresolved");
    }
    let mut l1 = z.clone();
    let mut l2 = z.clone();
    let mut signed = z.clone();
    let mut weight_sum = z.clone();
    for (d, w) in d.iter().zip(&w) {
        l1 = l1.add(&abs(d)?.mul(w));
        l2 = l2.add(&d.square().mul(w));
        signed = signed.add(&d.mul(w));
        weight_sum = weight_sum.add(w);
    }
    if !narrow(&[l1.clone(), signed.clone()], l1.upper(), requested)?
        || !narrow(std::slice::from_ref(&l2), l2.upper(), requested)?
    {
        return Ok(None);
    }
    let mut values = BTreeMap::new();
    values.insert("source_raw_center".into(), scale(&c, source_exp)?);
    values.insert("weighted_l1".into(), scale(&l1, de + we)?);
    values.insert("weighted_l2_squared".into(), scale(&l2, 2 * de + we)?);
    values.insert("signed_integral".into(), scale(&signed, de + we)?);
    let mut result = Measurement {
        values,
        fit_resolved: basis.is_empty(),
        arithmetic_precision: p,
    };
    if basis.is_empty() {
        return Ok(Some(result));
    }
    let cols = basis
        .iter()
        .map(|v| {
            normalized(
                &v.iter()
                    .map(|x| Ok(I::from_float(x, p)?))
                    .collect::<Result<Vec<_>>>()?,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let b = cols.len();
    let mut gram = vec![z.clone(); b * b];
    let mut rhs = vec![z.clone(); b];
    for a in 0..b {
        for j in 0..=n {
            rhs[a] = rhs[a].add(&cols[a].0[j].mul(&d[j]).mul(&w[j]));
        }
        result
            .values
            .insert(format!("rhs_{a}"), scale(&rhs[a], cols[a].1 + de + we)?);
        for k in 0..b {
            for (j, weight) in w.iter().enumerate() {
                gram[a * b + k] = gram[a * b + k].add(&cols[a].0[j].mul(&cols[k].0[j]).mul(weight));
            }
            result.values.insert(
                format!("gram_{a}_{k}"),
                scale(&gram[a * b + k], cols[a].1 + cols[k].1 + we)?,
            );
        }
    }
    if !narrow(&gram, weight_sum.upper(), requested)?
        || !narrow(&rhs, weight_sum.upper(), requested)?
    {
        return Ok(None);
    }
    let Some((fit, pivot)) = solve(&gram, &rhs, p)? else {
        return Ok(Some(result));
    };
    let fit_scale = fit
        .iter()
        .map(|x| abs(x).map(|x| x.upper().clone()))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .max_by(Float::total_cmp)
        .unwrap();
    if !narrow(&fit, &fit_scale, requested)? {
        return Ok(Some(result));
    }
    let mut residual = z.clone();
    for j in 0..=n {
        let mut rem = d[j].clone();
        for a in 0..b {
            rem = rem.sub(&fit[a].mul(&cols[a].0[j]));
        }
        residual = residual.add(&rem.square().mul(&w[j]));
    }
    // An exact zero optimum may have a small positive upper bound; retain it.
    if !narrow(std::slice::from_ref(&residual), l2.upper(), requested)? {
        return Ok(Some(result));
    }
    let fit = fit
        .iter()
        .zip(&cols)
        .map(|(x, (_, e))| scale(x, de - e))
        .collect::<Result<Vec<_>>>()?;
    if let Some(fixed) = fixed {
        let fixed = I::from_float(fixed, p)?;
        result
            .values
            .insert("b2".into(), fit[1].sub(&fixed.mul(&fit[0])));
        if !fit[0].contains_zero() {
            result
                .values
                .insert("b_effective".into(), fit[1].div(&fit[0])?);
        }
    }
    result.values.insert("minimum_scaled_pivot".into(), pivot);
    result.values.insert(
        "fit_residual_norm_squared".into(),
        scale(&residual, 2 * de + we)?,
    );
    for (a, x) in fit.into_iter().enumerate() {
        result.values.insert(format!("a_{a}"), x);
    }
    result.fit_resolved = true;
    Ok(Some(result))
}
pub(super) fn measure(
    v: &[Float],
    cutoff: &str,
    target: &[Float],
    basis: &[Vec<Float>],
    fixed: Option<&Float>,
    requested: u32,
) -> Result<Measurement> {
    if !(64..=1_000_000).contains(&requested)
        || v.is_empty()
        || v.len().is_multiple_of(2)
        || v.len() > 16385
        || v.iter().all(Float::is_zero)
        || v.iter().zip(v.iter().rev()).any(|(a, b)| a != b)
        || !(9..=131073).contains(&target.len())
        || target[0] != 1
        || basis.len() > 8
        || basis.iter().any(|b| b.len() != target.len())
        || (fixed.is_some() && basis.len() != 2)
        || v.iter()
            .chain(target)
            .chain(basis.iter().flatten())
            .chain(fixed)
            .any(|x| !x.is_finite() || x.prec() > 1_000_000)
    {
        bail!("invalid finite weighted profile shape, domain, normalization or precision");
    }
    let base = v
        .iter()
        .chain(target)
        .chain(basis.iter().flatten())
        .chain(fixed)
        .map(Float::prec)
        .fold(requested, u32::max);
    let mut last = None;
    for guard in [64u32, 128, 256, 512, 1024, 2048, 4096] {
        if let Some(result) = calculate(v, cutoff, target, basis, fixed, requested, base + guard)? {
            if result.fit_resolved {
                return Ok(result);
            }
            last = Some(result);
        }
    }
    last.ok_or_else(|| {
        anyhow::anyhow!("weighted profile arithmetic unresolved within 4096 guard bits")
    })
}
