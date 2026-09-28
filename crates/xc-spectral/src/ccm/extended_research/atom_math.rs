//! Finite stored-point atom sums. No omitted-tail or displacement assertion.
use anyhow::{bail, Result};
use rug::{float::Round, Float};
use std::cmp::Ordering;
use xc_numerics::mpfr_interval::MpfrInterval as I;

#[derive(Clone, Debug)]
pub(crate) struct Atom {
    pub coordinate: Float,
    pub weight: Float,
}
#[derive(Clone, Debug)]
pub(crate) struct Tail {
    pub cutoff: Float,
    pub count: usize,
    pub mass: Float,
    pub absolute_mass: Float,
    pub remaining: Float,
    pub moments: Option<[Float; 3]>,
    pub reason: Option<String>,
    pub arithmetic_precision: u32,
}
#[derive(Clone, Debug)]
pub(crate) struct Kernel {
    pub signed: [Float; 3],
    pub absolute: [Float; 3],
    pub closest: Option<Float>,
    pub arithmetic_precision: u32,
}
struct Group {
    x: Float,
    w: Float,
    absolute: Float,
    count: usize,
    origin_pole: bool,
}
/// Conservative per-calculation additional scratch estimate, not host memory.
pub(crate) fn scratch_bytes(atoms: &[Atom], rows: usize, requested: u32) -> Result<u64> {
    check(atoms, requested)?;
    let max = atoms
        .iter()
        .filter_map(|a| a.weight.get_exp())
        .max()
        .unwrap_or(0);
    let min = atoms
        .iter()
        .filter_map(|a| a.weight.get_exp())
        .min()
        .unwrap_or(max);
    let span = u64::try_from(i64::from(max) - i64::from(min))?;
    let bits = u64::from(requested) + span.min(1_000_000) + 4096 + 64;
    (atoms.len() as u64 + rows as u64 + 64)
        .checked_mul(36)
        .and_then(|x| x.checked_mul(bits.div_ceil(8) + 128))
        .ok_or_else(|| anyhow::anyhow!("atom scratch estimate overflow"))
}
fn check(atoms: &[Atom], requested: u32) -> Result<()> {
    if !(64..=1_000_000).contains(&requested)
        || atoms.iter().any(|a| {
            !a.coordinate.is_finite()
                || a.coordinate < 0
                || !a.weight.is_finite()
                || a.coordinate.prec() > requested
                || a.weight.prec() > requested
        })
    {
        bail!("invalid stored atom domain or requested precision");
    }
    Ok(())
}
fn output(x: &Float, p: u32) -> Result<Float> {
    let result = Float::with_val(p, x);
    if !result.is_finite() || (!x.is_zero() && result.is_zero()) {
        bail!("finite atom output exceeds exponent range");
    }
    Ok(result)
}
fn exact_add(a: &Float, b: &Float, p: u32) -> Result<Float> {
    let (value, dir) = Float::with_val_round(p, a + b, Round::Nearest);
    if !value.is_finite() || dir != Ordering::Equal {
        bail!("finite atom exact accumulation exceeds exponent range");
    }
    Ok(value)
}
// Every stored weight lies on a binary grid >= min_exp-max_precision.
// The absolute sum's exponent is <= max_exp+ceil(log2(count))+1.
fn groups(atoms: &[Atom], requested: u32) -> Result<(Vec<Group>, u32)> {
    check(atoms, requested)?;
    let max = atoms
        .iter()
        .filter_map(|a| a.weight.get_exp())
        .max()
        .unwrap_or(0);
    let min = atoms
        .iter()
        .filter_map(|a| a.weight.get_exp())
        .min()
        .unwrap_or(max);
    let span = i64::from(max) - i64::from(min);
    if span > 1_000_000 {
        bail!("exact atom accumulation exceeds one-million-bit exponent-span budget");
    }
    let source = atoms.iter().map(|a| a.weight.prec()).max().unwrap_or(64);
    let count_bits = usize::BITS - atoms.len().saturating_sub(1).leading_zeros();
    let p = u32::try_from(span)? + source + count_bits + 4;
    let mut sorted = atoms.to_vec();
    sorted.sort_by(|a, b| a.coordinate.total_cmp(&b.coordinate));
    let mut result: Vec<Group> = Vec::new();
    for a in sorted {
        if let Some(g) = result.last_mut().filter(|g| g.x == a.coordinate) {
            g.w = exact_add(&g.w, &a.weight, p)?;
            g.absolute = exact_add(&g.absolute, &a.weight.clone().abs(), p)?;
            g.count += 1;
            g.origin_pole |= a.coordinate.is_zero() && !a.weight.is_zero();
        } else {
            result.push(Group {
                x: a.coordinate.clone(),
                w: Float::with_val(p, &a.weight),
                absolute: Float::with_val(p, a.weight.clone().abs()),
                count: 1,
                origin_pole: a.coordinate.is_zero() && !a.weight.is_zero(),
            });
        }
    }
    Ok((result, p))
}
fn resolved(x: &I, p: u32) -> Result<Option<Float>> {
    x.validate()?;
    let lo = Float::with_val(p, x.lower());
    let hi = Float::with_val(p, x.upper());
    if !lo.is_finite() || !hi.is_finite() {
        bail!("finite atom result exceeds exponent range");
    }
    if lo != hi || (lo.is_zero() && (x.lower() != &0 || x.upper() != &0)) {
        return Ok(None);
    }
    Ok(Some(lo))
}
fn resolve_three(values: &[I; 3], p: u32) -> Result<Option<[Float; 3]>> {
    let mut result = Vec::new();
    for x in values {
        let Some(x) = resolved(x, p)? else {
            return Ok(None);
        };
        result.push(x);
    }
    Ok(Some(result.try_into().unwrap()))
}
pub(crate) fn tail(atoms: &[Atom], cutoffs: &[Float], requested: u32) -> Result<Vec<Tail>> {
    if cutoffs
        .iter()
        .any(|c| !c.is_finite() || c < &0 || c.prec() > requested)
        || cutoffs.windows(2).any(|w| w[0] >= w[1])
    {
        bail!("finite atom cutoffs must be ordered nonnegative stored points");
    }
    let (groups, exact) = groups(atoms, requested)?;
    let mut total = Float::with_val(exact, 0);
    for g in &groups {
        total = exact_add(&total, &g.w, exact)?;
    }
    let mut mass = Float::with_val(exact, 0);
    let mut absolute = mass.clone();
    let mut count = 0;
    let mut at = 0;
    let mut pole = false;
    let mut result = Vec::new();
    for cutoff in cutoffs {
        while at < groups.len() && &groups[at].x <= cutoff {
            let g = &groups[at];
            mass = exact_add(&mass, &g.w, exact)?;
            absolute = exact_add(&absolute, &g.absolute, exact)?;
            count += g.count;
            pole |= g.origin_pole;
            at += 1;
        }
        let remaining = exact_add(&total, &(-mass.clone()), exact)?;
        result.push(Tail {
            cutoff: output(cutoff, requested)?,
            count,
            mass: output(&mass, requested)?,
            absolute_mass: output(&absolute, requested)?,
            remaining: output(&remaining, requested)?,
            moments: None,
            reason: pole.then(|| {
                "inverse moments undefined: included origin has a nonzero supplied weight".into()
            }),
            arithmetic_precision: exact,
        });
    }
    let base = requested.max(exact);
    for guard in [64u32, 128, 256, 512, 1024, 2048, 4096] {
        let p = base + guard;
        let mut sums = std::array::from_fn(|_| I::from_i64(0, p));
        let mut at = 0;
        for r in &mut result {
            while at < groups.len() && groups[at].x <= r.cutoff {
                let g = &groups[at];
                at += 1;
                if g.x.is_zero() || g.w.is_zero() {
                    continue;
                }
                let x = I::from_float(&g.x, p)?;
                let mut term = I::from_float(&g.w, p)?;
                for sum in &mut sums {
                    term = term.div(&x)?;
                    *sum = sum.add(&term);
                }
            }
            if r.reason.is_none() && r.moments.is_none() {
                if let Some(values) = resolve_three(&sums, requested)? {
                    r.moments = Some(values);
                    r.arithmetic_precision = p;
                }
            }
        }
        if result
            .iter()
            .all(|r| r.moments.is_some() || r.reason.is_some())
        {
            break;
        }
    }
    for r in &mut result {
        if r.moments.is_none() && r.reason.is_none() {
            r.reason = Some(
                "finite inverse moments unresolved within 4096 guard bits or exponent range".into(),
            );
            r.arithmetic_precision = base + 4096;
        }
    }
    Ok(result)
}
pub(crate) fn kernel(atoms: &[Atom], z: &Float, requested: u32) -> Result<Kernel> {
    if !z.is_finite() || z.prec() > requested {
        bail!("invalid stored atom evaluation point");
    }
    let (groups, exact) = groups(atoms, requested)?;
    let base = requested.max(exact);
    let mut closest: Option<Float> = None;
    let p = base + 64;
    let guard = (Float::with_val(p, z).abs() + 1u32) >> (requested - 32);
    for g in &groups {
        let (distance, dir) = Float::with_val_round(p, &g.x - z, Round::Nearest);
        if !distance.is_finite() || (distance.is_zero() && dir != Ordering::Equal) {
            bail!("atom separation exceeds exponent range");
        }
        let distance = distance.abs();
        if distance <= guard {
            bail!("coincident or precision-limited included atom; full kernel sums withheld");
        }
        if closest.as_ref().is_none_or(|x| distance < *x) {
            closest = Some(distance);
        }
    }
    for guard in [64u32, 128, 256, 512, 1024, 2048, 4096] {
        let p = base + guard;
        let zero = I::from_i64(0, p);
        let z = I::from_float(z, p)?;
        let mut sums = std::array::from_fn(|_| zero.clone());
        let mut absolute = std::array::from_fn(|_| zero.clone());
        for g in &groups {
            let dx = I::from_float(&g.x, p)?.sub(&z);
            let distance = if dx.upper() < &0 {
                zero.sub(&dx)
            } else {
                dx.clone()
            };
            let mut signed = I::from_float(&g.w, p)?;
            let mut abs = I::from_float(&g.absolute, p)?;
            for j in 0..3 {
                signed = signed.div(&dx)?;
                abs = abs.div(&distance)?;
                sums[j] = sums[j].add(&signed);
                absolute[j] = absolute[j].add(&abs);
            }
        }
        if let (Some(signed), Some(absolute)) = (
            resolve_three(&sums, requested)?,
            resolve_three(&absolute, requested)?,
        ) {
            return Ok(Kernel {
                signed,
                absolute,
                closest: closest.as_ref().map(|x| output(x, requested)).transpose()?,
                arithmetic_precision: p,
            });
        }
    }
    bail!("finite atom kernels unresolved within 4096 guard bits or exponent range")
}
