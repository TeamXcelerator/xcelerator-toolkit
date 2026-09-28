//! Correctly rounded average of two exact stored finite points.
use crate::ccm::retained_evidence::finite_math::{scale, scale_float};
use anyhow::{bail, Result};
use rug::{float::Round, Float};
use xc_numerics::mpfr_interval::MpfrInterval as I;

pub(super) fn average(a: &Float, b: &Float) -> Result<Float> {
    let p = a.prec().max(b.prec());
    if !a.is_finite() || !b.is_finite() || p > 1_000_000 {
        bail!("invalid symmetry averaging source or precision");
    }
    if a == b {
        return Ok(Float::with_val(p, a));
    }
    let e = a
        .get_exp()
        .into_iter()
        .chain(b.get_exp())
        .max()
        .map_or(0, i64::from);
    for guard in [64, 128, 256, 512, 1024, 2048, 4096] {
        let work = p + guard;
        let values = [scale_float(a, -e, work)?, scale_float(b, -e, work)?];
        let sum = I::new(
            Float::with_val_round(work, Float::sum(values.iter()), Round::Down).0,
            Float::with_val_round(work, Float::sum(values.iter()), Round::Up).0,
        )?;
        let average = scale(&sum, e - 1)?;
        let lo = Float::with_val(p, average.lower());
        let hi = Float::with_val(p, average.upper());
        if lo == hi
            && lo.is_finite()
            && (!lo.is_zero() || (average.lower().is_zero() && average.upper().is_zero()))
        {
            return Ok(lo);
        }
    }
    bail!("symmetry average unresolved within 4096 guard bits")
}
