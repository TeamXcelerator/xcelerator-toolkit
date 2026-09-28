//! Resource and exact-decimal boundaries for finite-source certificates.
use anyhow::{bail, Context, Result};
use rug::{ops::Pow, Float, Integer, Rational};

pub(in crate::ccm) fn source_budget(count: usize, p: u32) -> Result<()> {
    if !(32..=1_000_000).contains(&p)
        || count == 0
        || count > 8193
        || (count as u128) * (u128::from(p).div_ceil(8) + 160) * 8 > (8u128 << 30)
    {
        bail!("finite-source dimensions or precision exceed the supported workspace budget");
    }
    Ok(())
}
pub(in crate::ccm) fn shape(modes: usize, count: usize, p: u32) -> Result<()> {
    if modes.checked_mul(2).and_then(|n| n.checked_add(1)) != Some(count) {
        bail!("finite CCM source requires exactly 2N+1 weights");
    }
    source_budget(count, p)
}
pub(super) fn isolation(bits: u32, p: u32) -> Result<()> {
    if !(32..=1_000_000).contains(&p) || bits < 16 || bits >= p {
        bail!("isolation width bits must be at least 16 and below supported source precision");
    }
    Ok(())
}
pub(in crate::ccm) fn rational_budget<'a>(
    values: impl Iterator<Item = &'a Float>,
    count: usize,
) -> Result<()> {
    let mut bits = 0u128;
    for x in values {
        if !x.is_finite() {
            bail!("nonfinite exact source point");
        }
        bits += u128::from(x.prec())
            + u128::from(i64::from(x.get_exp().unwrap_or(0)).unsigned_abs())
            + 2;
    }
    // Bound denominator clearing, coefficient storage and intermediate copies
    // before converting a compact MPFR exponent into a large GMP integer.
    if bits > 67_108_864 || (8 * count as u128 + 32) * bits.div_ceil(8) > (8u128 << 30) {
        bail!("exact secular polynomial exceeds the rational workspace budget");
    }
    Ok(())
}
/// Bound a retained vector of independently converted dyadic points and the
/// temporary rational values used by each subsequent scalar calculation.
/// This is not the polynomial denominator-clearing budget above.
pub(in crate::ccm) fn rational_point_vector_budget<'a>(
    values: impl Iterator<Item = &'a Float>,
) -> Result<()> {
    if rational_point_vector_workspace(values)? > (8u128 << 30) {
        bail!("exact point vector exceeds the rational workspace budget");
    }
    Ok(())
}
pub(in crate::ccm) fn rational_point_vector_workspace<'a>(
    values: impl Iterator<Item = &'a Float>,
) -> Result<u128> {
    let mut total_bits = 0u128;
    let mut maximum_bits = 0u128;
    let mut count = 0u128;
    for x in values {
        if !x.is_finite() {
            bail!("nonfinite exact source point");
        }
        // A nontrivial p-bit dyadic near one has both a p-bit numerator
        // and a p-bit power-of-two denominator after reduction.
        let bits = 2 * u128::from(x.prec())
            + u128::from(i64::from(x.get_exp().unwrap_or(0)).unsigned_abs())
            + 2;
        if bits > 67_108_864 {
            bail!("exact source point exceeds the rational conversion budget");
        }
        total_bits += bits;
        maximum_bits = maximum_bits.max(bits);
        count += 1;
    }
    Ok(total_bits.div_ceil(8) + 64 * maximum_bits.div_ceil(8) + count * 96)
}
pub(in crate::ccm) fn decimal(text: &str) -> Result<Rational> {
    if text.len() > 1_048_576 {
        bail!("decimal boundary exceeds the input budget");
    }
    let canonical = xc_core::DecimalLiteral::new(text)?.canonical()?;
    let (mantissa, exponent) = canonical
        .as_str()
        .split_once('e')
        .map_or(Ok((canonical.as_str(), 0i64)), |(m, e)| {
            e.parse::<i64>().map(|e| (m, e))
        })?;
    if exponent.unsigned_abs() > 1_000_000 {
        bail!("decimal boundary exponent exceeds the rational budget");
    }
    let integer =
        Integer::from_str_radix(mantissa, 10).context("parse exact decimal significand")?;
    let power = Integer::from(10).pow(exponent.unsigned_abs() as u32);
    Ok(if exponent >= 0 {
        Rational::from(integer * power)
    } else {
        Rational::from((integer, power))
    })
}
pub(super) fn rational_text(text: &str) -> Result<Rational> {
    if text.len() > 2_097_152 {
        bail!("rational boundary exceeds the input budget");
    }
    Ok(Rational::from(
        Rational::parse(text).context("parse exact rational boundary")?,
    ))
}

/// RN(p)(2*pi/L) for the exact stored length. Integer mode products are a
/// subsequent point stage, shared by production roots and certificate replay.
pub(in crate::ccm) fn rounded_spacing(length: &Float, p: u32) -> Result<Float> {
    use xc_numerics::mpfr_interval::MpfrInterval as I;
    if !(32..=1_000_000).contains(&p)
        || !length.is_finite()
        || length <= &0
        || length.prec() > 1_000_000
    {
        bail!("invalid secular spacing source or precision");
    }
    for guard in [64, 128, 256, 512, 1024, 2048, 4096] {
        let work = p.max(length.prec()) + guard;
        let spacing = I::pi(work)
            .mul(&I::from_i64(2, work))
            .div(&I::from_float(length, work)?)?;
        let lo = Float::with_val(p, spacing.lower());
        let hi = Float::with_val(p, spacing.upper());
        if lo == hi && lo.is_finite() && lo > 0 {
            return Ok(lo);
        }
    }
    bail!("secular spacing rounding unresolved within guard budget")
}
