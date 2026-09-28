//! Directed aggregate prime matrix at the exact rational cutoff.
//! Output entries are rounded once; exact prime edges contribute zero.
use super::*;
use rug::float::Round;
use xc_numerics::mpfr_interval::MpfrInterval as I;

fn log_one_plus(delta: &Rational, p: u32) -> Result<I> {
    let mut lo = Float::with_val_round(p, delta, Round::Down).0;
    let mut hi = Float::with_val_round(p, delta, Round::Up).0;
    lo.ln_1p_round(Round::Down);
    hi.ln_1p_round(Round::Up);
    Ok(I::new(lo, hi)?)
}
fn rounded(value: &I, p: u32) -> Result<Option<Float>> {
    value.validate()?;
    let lo = Float::with_val(p, value.lower());
    let hi = Float::with_val(p, value.upper());
    Ok((lo == hi
        && lo.is_finite()
        && (!lo.is_zero() || (value.lower().is_zero() && value.upper().is_zero())))
    .then_some(lo))
}
// A rational cutoff can share a rational logarithm ratio with a prime power
// only when it is an integer power of that same prime. Detect the exact
// quarter-turn cases so identically zero sine/cosine terms stay exact.
fn exact_reduced(c: &ExactCutoff, prime: u64, k: u32) -> Option<Rational> {
    if c.value().denom() != &1 {
        return None;
    }
    let mut remaining = c.value().numer().to_u64()?;
    let mut exponent = 0;
    while remaining.is_multiple_of(prime) {
        remaining /= prime;
        exponent += 1;
    }
    (remaining == 1 && exponent >= k).then(|| Rational::from((exponent - k, exponent)))
}
fn trig(n: usize, reduced: &I, exact: Option<&Rational>, pi: &I, p: u32) -> (I, I) {
    if n == 0 {
        return (I::from_i64(0, p), I::from_i64(1, p));
    }
    if let Some(r) = exact {
        let turns = r.clone() * (4 * n);
        if turns.denom() == &1 {
            let quadrant = Integer::from(turns.numer() % 4u32).to_u32().unwrap();
            return (
                I::from_i64([0, 1, 0, -1][quadrant as usize], p),
                I::from_i64([1, 0, -1, 0][quadrant as usize], p),
            );
        }
    }
    let phase = pi.mul(&I::from_u64((2 * n) as u64, p)).mul(reduced);
    (phase.sin(), phase.cos())
}
pub(super) fn evaluate(
    c: &ExactCutoff,
    n: usize,
    p: u32,
    o: &ResearchAssemblyOptions,
) -> Result<Vec<Float>> {
    require_precision(p)?;
    let d = o.validate(c, n)?;
    let work_max = p + 4096;
    // Include output, directed generators, prime data and the sieve before allocation.
    let memory = (d as u128 * d as u128 + 8 * u128::from(c.prime_cutoff()) + 32 * d as u128)
        * (u128::from(work_max).div_ceil(8) + 64);
    if memory > 8u128 << 30 {
        bail!("aggregate prime assembly exceeds the 8 GiB workspace budget");
    }
    let events = super::super::try_prime_powers_up_to(c.prime_cutoff())?;
    for guard in [64, 128, 256, 512, 1024, 2048, 4096] {
        let work = p + guard;
        let pi = I::pi(work);
        let two = I::from_i64(2, work);
        let length = log_one_plus(&Rational::from(c.value() - 1), work)?;
        let mut sines = vec![I::from_i64(0, work); n + 1];
        let mut diagonal = sines.clone();
        for &(power, prime, k) in &events {
            let delta = (c.value().clone() - power) / power;
            if delta == 0 {
                continue;
            }
            let reduced = log_one_plus(&delta, work)?.div(&length)?;
            let exact = exact_reduced(c, prime, k);
            let weight = I::from_u64(prime, work)
                .ln()?
                .div(&I::from_u64(power, work).sqrt()?)?;
            for mode in 0..=n {
                let (sin, cos) = trig(mode, &reduced, exact.as_ref(), &pi, work);
                sines[mode] = sines[mode].add(&weight.mul(&sin));
                diagonal[mode] = diagonal[mode].add(&two.mul(&reduced).mul(&weight).mul(&cos));
            }
        }
        let signed = |mode: i64| {
            let v = sines[mode.unsigned_abs() as usize].clone();
            if mode < 0 {
                v.neg()
            } else {
                v
            }
        };
        let mut matrix = vec![Float::with_val(p, 0); d * d];
        let mut resolved = true;
        'rows: for row in 0..d {
            let a = row as i64 - n as i64;
            for col in row..d {
                let b = col as i64 - n as i64;
                let value = if a == b {
                    diagonal[a.unsigned_abs() as usize].clone()
                } else {
                    signed(a)
                        .sub(&signed(b))
                        .div(&pi.mul(&I::from_i64(a - b, work)))?
                };
                let Some(value) = rounded(&value, p)? else {
                    resolved = false;
                    break 'rows;
                };
                matrix[row * d + col] = value.clone();
                matrix[col * d + row] = value;
            }
        }
        if resolved {
            return Ok(matrix);
        }
    }
    bail!("aggregate prime assembly rounding unresolved within 4096 guard bits")
}
