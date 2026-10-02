//! Correctly rounded finite prime-response action at the exact stored length
//! and vector. Transcendental constants are enclosed, not rounded inputs.
use super::*;
use crate::ccm::retained_evidence::finite_math::{scale, scale_float};
use rug::Rational;
use xc_numerics::mpfr_interval::MpfrInterval as I;
fn rounded(value: &I, p: u32) -> Result<Option<Float>> {
    value.validate()?;
    let lo = Float::with_val(p, value.lower());
    let hi = Float::with_val(p, value.upper());
    Ok((lo == hi
        && lo.is_finite()
        && (!lo.is_zero() || (value.lower().is_zero() && value.upper().is_zero())))
    .then_some(lo))
}
fn coefficients(
    vector: &[Rational],
    n_modes: usize,
    row: usize,
) -> Result<(Vec<Rational>, Rational)> {
    let n = row as i64 - n_modes as i64;
    let k = n.unsigned_abs() as usize;
    let mut cosine = vec![Rational::from(0); n_modes + 1];
    cosine[k] += vector[row].clone() * 2;
    for (column, v) in vector.iter().enumerate() {
        let m = column as i64 - n_modes as i64;
        if m == n {
            continue;
        }
        let factor: Rational = v.clone() * 2i32 / (n - m);
        cosine[k] += factor.clone() * n;
        cosine[m.unsigned_abs() as usize] -= factor * m;
    }
    let sine = vector[row].clone() * (-4 * (k as i64));
    let bits = cosine
        .iter()
        .chain(std::iter::once(&sine))
        .map(|x| {
            u128::from(x.numer().significant_bits()) + u128::from(x.denom().significant_bits())
        })
        .sum::<u128>();
    if bits > 67_108_864 {
        bail!("prime-response row exceeds exact coefficient budget");
    }
    Ok((cosine, sine))
}
pub(super) fn evaluate(
    n_modes: usize,
    power: u64,
    prime: u64,
    l: &Float,
    vector: &[Float],
    p: u32,
    parallel: bool,
) -> Result<PrimePowerVelocityAction> {
    if !(64..=1_000_000).contains(&p)
        || n_modes > 4096
        || n_modes.checked_mul(2).and_then(|n| n.checked_add(1)) != Some(vector.len())
        || !l.is_finite()
        || l <= &0
        || l.prec() > 1_000_000
        || vector
            .iter()
            .any(|x| !x.is_finite() || x.prec() > 1_000_000)
        || power < 2
        || prime < 2
        || prime > power
    {
        bail!("invalid finite prime-response source, shape, event or precision");
    }
    let mut remaining = power;
    while remaining.is_multiple_of(prime) {
        remaining /= prime;
    }
    if remaining != 1 {
        bail!("prime-response power and prime do not agree");
    }
    let base = vector
        .iter()
        .map(Float::prec)
        .max()
        .unwrap_or(p)
        .max(l.prec())
        .max(p);
    let workers = if parallel {
        rayon::current_num_threads().min(vector.len())
    } else {
        1
    };
    let bytes = vector.len() as u128
        * (u128::from(base * 2 + 4096).div_ceil(8) + 192)
        * ((workers as u128 + 1) * 32);
    if bytes > super::source_working_budget()? {
        bail!("prime-response action exceeds workspace budget");
    }
    let exponent = vector
        .iter()
        .filter_map(Float::get_exp)
        .max()
        .map_or(0, i64::from);
    let normalized = vector
        .iter()
        .map(|x| scale_float(x, -exponent, base))
        .collect::<Result<Vec<_>>>()?;
    crate::ccm::certified_roots::boundary::rational_budget(normalized.iter(), normalized.len())?;
    let exact = normalized
        .iter()
        .map(|x| x.to_rational().expect("finite normalized source"))
        .collect::<Vec<_>>();
    // Double the source precision: at a stored approximation of log(power),
    // 1-log(power)/L can be about 2^-p and still needs p relative output bits.
    for guard in [64, 128, 256, 512, 1024, 2048, 4096] {
        let work = base * 2 + guard;
        let one = I::from_i64(1, work);
        let two = I::from_i64(2, work);
        let pi = I::pi(work);
        let length = I::from_float(l, work)?;
        let log_power = I::from_u64(power, work).ln()?;
        let weight = I::from_u64(prime, work).ln()?;
        let sqrt_power = I::from_u64(power, work).sqrt()?;
        let reduced = one.sub(&log_power.div(&length)?);
        let coefficient = weight
            .mul(&log_power)
            .div(&sqrt_power)?
            .div(&length.square())?
            .neg();
        let edge = weight.mul(&two).div(&sqrt_power)?.div(&log_power)?.neg();
        let metadata = [&log_power, &weight, &reduced, &coefficient, &edge]
            .into_iter()
            .map(|v| rounded(v, p))
            .collect::<Result<Option<Vec<_>>>>()?;
        let Some(metadata) = metadata else {
            continue;
        };
        let phase_scale = pi.mul(&two).mul(&reduced);
        let phases = (0..=n_modes)
            .map(|n| phase_scale.mul(&I::from_u64(n as u64, work)))
            .collect::<Vec<_>>();
        let cosines = phases.iter().map(I::cos).collect::<Vec<_>>();
        let sines = phases.iter().map(I::sin).collect::<Vec<_>>();
        let row = |index: usize| -> Result<Option<Float>> {
            let (cosine, sine) = coefficients(&exact, n_modes, index)?;
            let mut terms = Vec::with_capacity(cosine.len() + 1);
            for (factor, value) in cosine.iter().zip(&cosines) {
                if factor != &0 {
                    terms.push(I::from_rational(factor, work).mul(value));
                }
            }
            if sine != 0 {
                terms.push(
                    I::from_rational(&sine, work)
                        .mul(&pi)
                        .mul(&reduced)
                        .mul(&sines[index.abs_diff(n_modes)]),
                );
            }
            let sum = I::new(
                Float::with_val_round(
                    work,
                    Float::sum(terms.iter().map(I::lower)),
                    rug::float::Round::Down,
                )
                .0,
                Float::with_val_round(
                    work,
                    Float::sum(terms.iter().map(I::upper)),
                    rug::float::Round::Up,
                )
                .0,
            )?;
            rounded(&scale(&sum.mul(&coefficient), exponent)?, p)
        };
        let results = if parallel {
            (0..vector.len())
                .into_par_iter()
                .map(row)
                .collect::<Vec<_>>()
        } else {
            (0..vector.len()).map(row).collect::<Vec<_>>()
        };
        if let Some(action) = results.into_iter().collect::<Result<Option<Vec<_>>>>()? {
            return Ok(PrimePowerVelocityAction {
                log_power: metadata[0].clone(),
                von_mangoldt_weight: metadata[1].clone(),
                reduced_position: metadata[2].clone(),
                velocity_coefficient: metadata[3].clone(),
                edge_jump_coefficient: metadata[4].clone(),
                action,
            });
        }
    }
    bail!("prime-response action rounding unresolved within the guard budget")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn verified_prime_kernel_matches_independent_direct_high_precision_matrix() {
        for p in [64, 128, 256] {
            for n in 1..=3usize {
                for length in [3, 5, 7] {
                    for exponent in [-1000i32, 0, 1000] {
                        let vector = (0..2 * n + 1)
                            .map(|k| {
                                Float::with_val(p, (k as i32 - 2) * (k as i32 + 1)) << exponent
                            })
                            .collect::<Vec<_>>();
                        let l = Float::with_val(p, length);
                        let actual = evaluate(n, 4, 2, &l, &vector, p, true).unwrap();
                        let reference=super::super::response_performance_reference::direct_prime_power_velocity_reference(n,4,2,&Float::with_val(1024,&l),&vector.iter().map(|x|Float::with_val(1024,x)).collect::<Vec<_>>(),1024).unwrap();
                        for (i, (got, want)) in
                            actual.action.iter().zip(&reference.action).enumerate()
                        {
                            assert_eq!(
                                *got,
                                Float::with_val(p, want),
                                "p={p},n={n},L={length},e={exponent},row={i}"
                            );
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn exact_odd_cancellation_and_zero_vector_remain_exact() {
        let p = 128;
        let l = Float::with_val(p, 3);
        let a = Float::with_val(p, 1) << 400u32;
        let odd = evaluate(
            1,
            2,
            2,
            &l,
            &[a.clone(), Float::with_val(p, 0), -a],
            p,
            true,
        )
        .unwrap();
        assert!(odd.action[1].is_zero());
        assert_eq!(odd.action[0], -odd.action[2].clone());
        let zero = evaluate(2, 8, 2, &l, &vec![Float::with_val(p, 0); 5], p, true).unwrap();
        assert!(zero.action.iter().all(Float::is_zero));
        assert!(evaluate(0, 8, 3, &l, &[Float::with_val(p, 1)], p, true).is_err());
        for bad in [0, 63, 1_000_001, u32::MAX] {
            assert!(evaluate(0, 2, 2, &l, &[Float::with_val(p, 1)], bad, true).is_err());
        }
    }
}
