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

/// Independently converted source points and the live exact row scratch used
/// by finite prime/u-flow actions. The reservation outlives the rational data.
pub(in crate::ccm) struct AdmittedResponsePoints {
    pub values: Vec<Rational>,
    pub workspace_bytes: u128,
    _reservation: Option<xc_core::MemoryReservation>,
}

pub(in crate::ccm) fn response_point_workspace(
    values: &[Float],
    modes: usize,
    workers: usize,
) -> Result<u128> {
    let overflow = || anyhow::anyhow!("exact response point workspace overflow");
    if modes > 4096
        || modes.checked_mul(2).and_then(|n| n.checked_add(1)) != Some(values.len())
        || workers == 0
    {
        bail!("invalid exact response point shape or worker allowance");
    }
    // Keep the independent conversion estimator and its per-point exponent
    // guard. No numerator polynomial or denominator product is constructed.
    let points = rational_point_vector_workspace(values.iter())?;
    let maximum_bits = values
        .iter()
        .map(|x| {
            2 * u128::from(x.prec())
                + u128::from(i64::from(x.get_exp().unwrap_or(0)).unsigned_abs())
                + 2
        })
        .max()
        .unwrap_or(0);
    let distance = modes * 2;
    let distance_bits = usize::BITS - distance.max(1).leading_zeros();
    let lcm_bits = (distance as u128) * u128::from(distance_bits) + 1;
    let count_bits = usize::BITS - values.len().max(1).leading_zeros();
    let upper_exponent = values
        .iter()
        .filter_map(Float::get_exp)
        .map(i64::from)
        .max()
        .unwrap_or(0)
        .max(0) as u128;
    // All points are dyadic. Every row denominator divides 2^D*lcm(1..2N),
    // with D <= maximum_bits and |value| < 2^upper_exponent;
    // multiplication by mode indices and addition of
    // at most 2N+1 terms add at most two count-bit allowances plus constants.
    // The following bound covers numerator AND denominator bits, including
    // intermediate scalar factors. It applies to both original rational rows
    // and the optional common-denominator construction.
    let component_bits = maximum_bits
        .checked_add(lcm_bits)
        .and_then(|n| n.checked_add(2 * u128::from(count_bits) + 8))
        .and_then(|n| n.checked_mul(2))
        .and_then(|n| n.checked_add(upper_exponent))
        .ok_or_else(overflow)?;
    // Adapt the established 64-scalar conversion/arithmetic scratch allowance
    // to each live row component. This covers the row's rational storage,
    // integer/rational construction overlap and native scalar temporaries.
    // Rows never dispatch nested Rayon work; their live count is pool-bounded.
    let row = component_bits
        .div_ceil(8)
        .checked_add(96)
        .and_then(|n| n.checked_mul(64))
        .and_then(|n| n.checked_mul((modes + 2) as u128))
        .and_then(|n| n.checked_mul(workers.min(values.len()) as u128))
        .ok_or_else(overflow)?;
    points.checked_add(row).ok_or_else(overflow)
}

/// Admit before `to_rational`, then retain the owner across every exact row
/// using these points. Other MPFR/table workspace keeps its existing estimate.
pub(in crate::ccm) fn response_points(
    values: &[Float],
    modes: usize,
    workers: usize,
    other_workspace_bytes: u128,
    maximum_bytes: u128,
) -> Result<AdmittedResponsePoints> {
    let workspace_bytes = response_point_workspace(values, modes, workers)?;
    let resources = xc_core::RunResources::current();
    let limit = maximum_bytes.min(
        resources
            .as_ref()
            .map_or(8u128 << 30, |r| u128::from(r.maximum_bytes())),
    );
    if workspace_bytes
        .checked_add(other_workspace_bytes)
        .is_none_or(|bytes| bytes > limit)
    {
        bail!("exact response point conversion exceeds the rational workspace budget: exact_bytes={workspace_bytes}, other_bytes={other_workspace_bytes}, maximum_working_bytes={limit}");
    }
    let reservation =
        resources
            .as_ref()
            .map(|r| {
                r.try_reserve_working(workspace_bytes.try_into().map_err(|_| {
                    xc_core::ConfigError::new("exact response workspace exceeds u64")
                })?)
            })
            .transpose()?;
    let mut exact = Vec::with_capacity(values.len());
    for value in values {
        exact.push(
            value
                .to_rational()
                .context("convert admitted finite response point")?,
        );
    }
    Ok(AdmittedResponsePoints {
        values: exact,
        workspace_bytes,
        _reservation: reservation,
    })
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

#[cfg(test)]
mod capacity_tests {
    use super::*;

    #[test]
    fn capacity_response_vector_does_not_charge_a_nonexistent_polynomial() {
        let p = 6708;
        let value = Float::with_val(p, 1) - (Float::with_val(p, 1) >> p);
        let values = vec![value; 2001];
        // About 3.4 MB of exact input limbs. The old polynomial-denominator
        // screen charges this independent vector as a quadratic polynomial.
        assert!(rational_budget(values.iter(), values.len()).is_err());
        let resources = xc_core::RunResources::with_worker_limit(16 << 30, 2).unwrap();
        let _scope = resources.enter();
        let expected = response_point_workspace(&values, 1000, 2).unwrap();
        assert!(expected < 8u128 << 30);
        {
            let exact = response_points(&values, 1000, 2, 0, 16u128 << 30).unwrap();
            assert_eq!(exact.values.len(), values.len());
            assert!(exact
                .values
                .iter()
                .all(|v| v == &values[0].to_rational().unwrap()));
            assert_eq!(u128::from(resources.usage().working_bytes), expected);
        }
        assert_eq!(resources.usage().working_bytes, 0);
        assert_eq!(resources.usage().retained_bytes, 0);
        assert_eq!(u128::from(resources.usage().peak_bytes), expected);
    }

    #[test]
    fn capacity_response_points_share_remaining_budget_and_release_on_drop() {
        let values = [1, -2, 3].map(|n| Float::with_val(128, n));
        let bytes = response_point_workspace(&values, 1, 2).unwrap() as u64;
        let resources = xc_core::RunResources::new(bytes + 1024).unwrap();
        let _scope = resources.enter();
        let parent = resources.try_reserve_retained(512).unwrap();
        let before = resources.usage();
        let error = response_points(&values, 1, 2, 0, u128::from(bytes - 1))
            .err()
            .expect("declared stage budget must refuse before conversion");
        assert!(error
            .to_string()
            .contains("exceeds the rational workspace budget"));
        assert_eq!(resources.usage(), before);
        {
            let exact =
                response_points(&values, 1, 2, 128, u128::from(resources.maximum_bytes())).unwrap();
            assert_eq!(
                exact.values,
                vec![Rational::from(1), Rational::from(-2), Rational::from(3)]
            );
            assert_eq!(resources.usage().working_bytes, bytes);
        }
        assert_eq!(resources.usage().working_bytes, 0);
        let occupied = resources.try_reserve_retained(513).unwrap();
        let before = resources.usage();
        assert!(response_points(&values, 1, 2, 0, u128::from(resources.maximum_bytes())).is_err());
        assert_eq!(resources.usage(), before);
        drop(occupied);
        drop(parent);
        assert_eq!(resources.usage().retained_bytes, 0);
    }

    #[test]
    fn capacity_response_points_refuse_compact_extreme_exponents_before_conversion() {
        let huge = Float::with_val(128, 1) >> 700_000_000u32;
        let values = [huge, Float::with_val(128, 0), Float::with_val(128, 1)];
        assert!(response_point_workspace(&values, 1, 1).is_err());
        assert!(response_point_workspace(&values, usize::MAX, 1).is_err());
        assert!(response_point_workspace(&values, 1, 0).is_err());
        // Metadata alone exceeds this real owner's capacity, even though the
        // compact source Float itself needs only a few limbs.
        let wide = Float::with_val(128, 1) >> 1_000_000u32;
        let values = [wide, Float::with_val(128, 0), Float::with_val(128, 1)];
        let resources = xc_core::RunResources::new(1024).unwrap();
        let _scope = resources.enter();
        assert!(response_points(&values, 1, 1, 0, 1024).is_err());
        assert_eq!(resources.usage().peak_bytes, 0);
    }
}
