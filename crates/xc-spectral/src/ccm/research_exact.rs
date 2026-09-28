//! Bounded exact stored-point algebra for finite research diagnostics.
use anyhow::{bail, Result};
use rug::{Float, Rational};
use xc_numerics::mpfr_interval::MpfrInterval as I;
pub(super) fn budget<'a>(values: impl Iterator<Item = &'a Rational>) -> Result<()> {
    let bits = values
        .map(|x| {
            u128::from(x.numer().significant_bits()) + u128::from(x.denom().significant_bits())
        })
        .sum::<u128>();
    if bits > 67_108_864 {
        bail!("exact research algebra exceeds its 64 Mbit rational budget");
    }
    Ok(())
}
pub(super) fn points(values: &[Float], p: u32, max_count: usize) -> Result<Vec<Rational>> {
    if !(64..=1_000_000).contains(&p)
        || values.is_empty()
        || values.len() > max_count
        || values
            .iter()
            .any(|x| !x.is_finite() || x.prec() < p || x.prec() > 1_000_000)
    {
        bail!("invalid exact research source or precision");
    }
    super::super::certified_roots::boundary::rational_point_vector_budget(values.iter())?;
    Ok(values
        .iter()
        .map(|x| x.to_rational().expect("validated finite source"))
        .collect())
}
pub(super) fn point(value: &Float, p: u32) -> Result<Rational> {
    Ok(points(std::slice::from_ref(value), p, 1)?.pop().unwrap())
}
pub(super) fn output(value: &Rational, p: u32) -> Result<Float> {
    budget(std::iter::once(value))?;
    let out = Float::with_val(p, value);
    if !out.is_finite() || (out.is_zero() && value != &0) {
        bail!("exact research result exceeds MPFR output range");
    }
    Ok(out)
}
pub(super) fn sqrt(value: &Rational, p: u32) -> Result<Float> {
    budget(std::iter::once(value))?;
    if value < &0 {
        bail!("negative research squared norm");
    }
    for guard in [64, 128, 256, 512, 1024, 2048, 4096] {
        let v = I::from_rational(value, p + guard).sqrt()?;
        let lo = Float::with_val(p, v.lower());
        let hi = Float::with_val(p, v.upper());
        if lo == hi && lo.is_finite() && (!lo.is_zero() || value == &0) {
            return Ok(lo);
        }
    }
    bail!("research norm rounding unresolved within 4096 guard bits")
}
pub(super) struct Schur {
    pub defect: Rational,
    pub border_squared: Rational,
    pub value: Rational,
}
pub(super) fn schur(
    small: &[Rational],
    large: &[Rational],
    n: usize,
    shift: &Rational,
) -> Result<Schur> {
    if n == 0
        || n > 256
        || n.checked_mul(n) != Some(small.len())
        || (n + 1).checked_mul(n + 1) != Some(large.len())
    {
        bail!("exact Schur diagnostics require square inputs and 1..256 prefix dimension");
    }
    budget(small.iter().chain(large).chain(std::iter::once(shift)))?;
    let m = n + 1;
    let mut defect = Rational::from(0);
    let mut block = Vec::with_capacity(n * n);
    let mut border = Vec::with_capacity(n);
    for i in 0..n {
        border.push(large[i * m + n].clone());
        for j in 0..n {
            defect = defect.max((large[i * m + j].clone() - &small[i * n + j]).abs());
            block.push(
                large[i * m + j].clone()
                    - if i == j {
                        shift.clone()
                    } else {
                        Rational::from(0)
                    },
            );
        }
    }
    let mut a = block.clone();
    let mut rhs = border.clone();
    budget(a.iter().chain(&rhs))?;
    // Exact Gaussian elimination with row pivoting. Every update is rational;
    // a zero pivot after searching the column means the block is singular.
    for k in 0..n {
        let row = (k..n)
            .find(|i| a[i * n + k] != 0)
            .ok_or_else(|| anyhow::anyhow!("singular exact Schur prefix"))?;
        if row != k {
            for j in 0..n {
                a.swap(k * n + j, row * n + j);
            }
            rhs.swap(k, row);
        }
        for i in k + 1..n {
            let factor = a[i * n + k].clone() / &a[k * n + k];
            for j in k + 1..n {
                let term = factor.clone() * &a[k * n + j];
                a[i * n + j] -= term;
            }
            let term = factor * &rhs[k];
            rhs[i] -= term;
            a[i * n + k] = Rational::from(0);
            budget(a[i * n..(i + 1) * n].iter().chain(std::iter::once(&rhs[i])))?;
        }
        budget(a.iter().chain(&rhs))?;
    }
    let mut solution = vec![Rational::from(0); n];
    for i in (0..n).rev() {
        let mut value = rhs[i].clone();
        for j in i + 1..n {
            value -= a[i * n + j].clone() * &solution[j];
            budget(std::iter::once(&value))?;
        }
        solution[i] = value / &a[i * n + i];
        budget(solution.iter())?;
    }
    // Check the original system independently of the elimination state.
    for i in 0..n {
        let mut value = Rational::from(0);
        for j in 0..n {
            value += block[i * n + j].clone() * &solution[j];
            budget(std::iter::once(&value))?;
        }
        if value != border[i] {
            bail!("exact Schur solution failed original-system substitution");
        }
    }
    let mut value = large[n * m + n].clone() - shift;
    let mut border_squared = Rational::from(0);
    for (b, x) in border.iter().zip(&solution) {
        value -= b.clone() * x;
        border_squared += b.clone() * b;
        budget([&value, &border_squared].into_iter())?;
    }
    Ok(Schur {
        defect,
        border_squared,
        value,
    })
}
pub(super) fn pencil(a: &[Rational], g: &[Rational], z: &Rational) -> Result<Vec<Rational>> {
    if a.len() != g.len() {
        bail!("research pencil shape mismatch");
    }
    let mut bits = 0u128;
    let mut out = Vec::with_capacity(a.len());
    for (a, g) in a.iter().zip(g) {
        let value = a.clone() - g.clone() * z;
        bits += u128::from(value.numer().significant_bits())
            + u128::from(value.denom().significant_bits());
        if bits > 67_108_864 {
            bail!("exact pencil exceeds rational budget");
        }
        out.push(value);
    }
    Ok(out)
}
pub(super) struct Secular {
    pub value: Rational,
    pub derivative: Rational,
    pub nearest: Rational,
    pub absolute_sum: Rational,
}
pub(super) fn secular(
    weights: &[Rational],
    poles: &[Rational],
    point: &Rational,
) -> Result<Secular> {
    if weights.is_empty()
        || weights.len() != poles.len()
        || weights.len() > 8193
        || poles.windows(2).any(|p| p[0] >= p[1])
    {
        bail!("invalid exact secular source");
    }
    let mut value = Rational::from(0);
    let mut derivative = Rational::from(0);
    let mut absolute_sum = Rational::from(0);
    let mut nearest: Option<Rational> = None;
    for (w, pole) in weights.iter().zip(poles) {
        let den = point.clone() - pole;
        if den == 0 {
            bail!("secular evaluation encountered a pole");
        }
        nearest = Some(nearest.map_or_else(|| den.clone().abs(), |old| old.min(den.clone().abs())));
        let term = w.clone() / &den;
        value += &term;
        absolute_sum += term.clone().abs();
        derivative -= term / den;
        budget([&value, &derivative, &absolute_sum].into_iter())?;
    }
    Ok(Secular {
        value,
        derivative,
        nearest: nearest.unwrap(),
        absolute_sum,
    })
}
