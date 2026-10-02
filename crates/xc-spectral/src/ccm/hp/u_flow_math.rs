//! Directed analytic u-flow on the exact stored length, vector, and quadrature
//! tables. This verifies finite arithmetic, not quadrature truncation error.
#[cfg(test)]
#[path = "u_flow.rs"]
mod u_flow;

use super::*;
use crate::ccm::retained_evidence::finite_math::{scale, scale_float};
use rug::{float::Round, Rational};
use xc_numerics::mpfr_interval::MpfrInterval as I;
type Table = (Vec<Float>, Vec<Float>);
fn sum(values: &[I], p: u32) -> Result<I> {
    for v in values {
        v.validate()?;
    }
    Ok(I::new(
        Float::with_val_round(p, Float::sum(values.iter().map(I::lower)), Round::Down).0,
        Float::with_val_round(p, Float::sum(values.iter().map(I::upper)), Round::Up).0,
    )?)
}
fn rounded(v: &I, p: u32) -> Result<Option<Float>> {
    v.validate()?;
    let a = Float::with_val(p, v.lower());
    let b = Float::with_val(p, v.upper());
    Ok(
        (a == b && a.is_finite() && (!a.is_zero() || (v.lower().is_zero() && v.upper().is_zero())))
            .then_some(a),
    )
}
fn sinh(v: &I) -> Result<I> {
    v.validate()?;
    let mut a = v.lower().clone();
    let mut b = v.upper().clone();
    a.sinh_round(Round::Down);
    b.sinh_round(Round::Up);
    Ok(I::new(a, b)?)
}
fn exp_m1(v: &I) -> Result<I> {
    v.validate()?;
    let mut a = v.lower().clone();
    let mut b = v.upper().clone();
    a.exp_m1_round(Round::Down);
    b.exp_m1_round(Round::Up);
    Ok(I::new(a, b)?)
}
fn exact_quadrant(node: &Float, n: usize) -> Result<Option<usize>> {
    // Only one converted node is retained at a time. A polynomial budget
    // across the full rule would charge for nonexistent coefficient storage.
    crate::ccm::certified_roots::boundary::rational_budget(std::iter::once(node), 1)?;
    let twice_mode = n
        .checked_mul(2)
        .ok_or_else(|| anyhow::anyhow!("u-flow mode overflow"))?;
    let exact_turn: Rational =
        (node.to_rational().expect("validated finite node") + 1i32) * twice_mode;
    Ok(if exact_turn.denom() == &1 {
        exact_turn
            .numer()
            .to_i64()
            .map(|q| q.rem_euclid(4) as usize)
    } else {
        None
    })
}
fn integral(n: usize, l: &I, nodes: &[Float], weights: &[Float], p: u32) -> Result<[I; 3]> {
    if nodes.len() != weights.len()
        || nodes.is_empty()
        || nodes.len() > 1_000_000
        || nodes.iter().any(|x| !x.is_finite() || x <= &-1 || x >= &1)
        || weights.iter().any(|x| !x.is_finite() || x <= &0)
    {
        bail!("invalid stored u-flow quadrature rule");
    }
    let one = I::from_i64(1, p);
    let two = I::from_i64(2, p);
    let half = I::point(Float::with_val(p, 0.5));
    let pi = I::pi(p);
    let mut alpha = Vec::with_capacity(nodes.len());
    let mut beta = alpha.clone();
    let mut gamma = alpha.clone();
    for (node, weight) in nodes.iter().zip(weights) {
        let h = I::from_float(node, p)?.add(&one).mul(&half);
        let x = l.mul(&h);
        let e = x.mul(&half).neg().exp();
        let rho = e.div(&exp_m1(&x.mul(&two).neg())?.neg())?;
        let coth = one.add(&two.div(&exp_m1(&x.mul(&two))?)?);
        let combined = rho.mul(&one.add(&x.mul(&half.sub(&coth))));
        let phase = pi.mul(&I::from_u64((2 * n) as u64, p)).mul(&h);
        // A tiny nonzero stored node can disappear in rounded `node + 1`.
        // Only the exact dyadic phase may establish a trigonometric identity.
        let quadrant = exact_quadrant(node, n)?;
        let (sin, cos, difference_cos) = if let Some(q) = quadrant {
            let cos = I::from_i64([1, 0, -1, 0][q], p);
            (I::from_i64([0, 1, 0, -1][q], p), cos.clone(), cos.sub(&one))
        } else {
            (
                phase.sin(),
                phase.cos(),
                phase.mul(&half).sin().square().mul(&two).neg(),
            )
        };
        let weight = I::from_float(weight, p)?;
        alpha.push(weight.mul(&sin).mul(&combined));
        beta.push(weight.mul(&h).mul(&cos).mul(&combined));
        let difference = difference_cos.sub(&exp_m1(&x.mul(&half).neg())?);
        gamma.push(
            weight.mul(
                &difference
                    .mul(&combined)
                    .add(&x.mul(&e).mul(&rho).mul(&half)),
            ),
        );
    }
    Ok([
        sum(&alpha, p)?.div(&two.mul(&pi))?,
        sum(&beta, p)?.mul(&half),
        sum(&gamma, p)?
            .mul(&half)
            .add(&one.div(&two.mul(&sinh(l)?))?),
    ])
}
fn arch_coefficients(v: &[Rational], n: usize, row: usize) -> (Vec<Rational>, Rational) {
    let a = row as i64 - n as i64;
    let mut alpha = vec![Rational::from(0); n + 1];
    for (col, value) in v.iter().enumerate() {
        let b = col as i64 - n as i64;
        if a == b {
            continue;
        }
        let t = value.clone() / (a - b);
        alpha[b.unsigned_abs() as usize] -= t.clone() * b.signum();
        alpha[a.unsigned_abs() as usize] += t * a.signum();
    }
    (alpha, v[row].clone() * 2)
}
fn pole(v: &[Rational], n: usize, l: &I, p: u32) -> Result<Vec<I>> {
    let two = I::from_i64(2, p);
    let four = I::from_i64(4, p);
    let c = I::pi(p).square().mul(&I::from_i64(16, p));
    let l2 = l.square();
    let quarter = sinh(&l.div(&four)?)?.square();
    let half = sinh(&l.div(&two)?)?;
    let mut d = Vec::with_capacity(n + 1);
    let mut evens = Vec::with_capacity(n + 1);
    let mut odds = Vec::with_capacity(n + 1);
    for k in 0..=n {
        d.push(l2.add(&c.mul(&I::from_u64((k * k) as u64, p))));
        if k == 0 {
            evens.push(v[n].clone());
            odds.push(Rational::from(0));
        } else {
            evens.push(v[n + k].clone() + &v[n - k]);
            odds.push((v[n + k].clone() - &v[n - k]) * k);
        }
    }
    let mut even = Vec::new();
    let mut odd = Vec::new();
    let mut even_derivative = Vec::new();
    let mut odd_derivative = Vec::new();
    for k in 0..=n {
        let a = I::from_rational(&evens[k], p).div(&d[k])?;
        let b = I::from_rational(&odds[k], p).div(&d[k])?;
        even_derivative.push(a.div(&d[k])?);
        odd_derivative.push(b.div(&d[k])?);
        even.push(a);
        odd.push(b);
    }
    let s0 = sum(&even, p)?;
    let s1 = sum(&odd, p)?;
    let s0d = sum(&even_derivative, p)?.mul(l).mul(&two).neg();
    let s1d = sum(&odd_derivative, p)?.mul(l).mul(&two).neg();
    let prefactor = l.mul(&quarter).mul(&I::from_i64(32, p));
    let prefactor_derivative = quarter
        .add(&l.mul(&half).div(&four)?)
        .mul(&I::from_i64(32, p));
    (0..v.len())
        .map(|row| {
            let a = row as i64 - n as i64;
            let denominator = &d[a.unsigned_abs() as usize];
            let f = prefactor.div(denominator)?;
            let fd = prefactor_derivative
                .div(denominator)?
                .sub(&f.mul(l).mul(&two).div(denominator)?);
            let signed = c.mul(&I::from_i64(a, p));
            let numerator = l2.mul(&s0).sub(&signed.mul(&s1));
            let derivative = two
                .mul(l)
                .mul(&s0)
                .add(&l2.mul(&s0d))
                .sub(&signed.mul(&s1d));
            Ok(fd.mul(&numerator).add(&f.mul(&derivative)))
        })
        .collect()
}
fn rounding_enclosure(v: Float) -> Result<I> {
    if v.is_zero() {
        return Ok(I::point(v));
    }
    let mut lo = v.clone();
    let mut hi = v;
    lo.next_down();
    hi.next_up();
    Ok(I::new(lo, hi)?)
}
pub(super) fn evaluate(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    l: &Float,
    vector: &[Float],
) -> Result<UFlowVelocityActions> {
    let p = cfg.precision_bits;
    let n = params.n_modes;
    if !(64..=995_904).contains(&p)
        || n > 4096
        || n.checked_mul(2).and_then(|x| x.checked_add(1)) != Some(vector.len())
        || !l.is_finite()
        || l <= &0
        || l.prec() > 995_904
        || vector.iter().any(|v| !v.is_finite() || v.prec() > 995_904)
        || cfg.quad_points == 0
        || cfg.quad_points > 1_000_000
        || params.lambda_sq_int() > 10_000_000
    {
        bail!("invalid or unsupported u-flow source, precision, shape or resource request");
    }
    let base = vector
        .iter()
        .map(Float::prec)
        .max()
        .unwrap_or(p)
        .max(p)
        .max(l.prec());
    let orders = (0..=n)
        .map(|mode| cfg.quad_points.max(3 * mode + (p / 2) as usize))
        .collect::<Vec<_>>();
    let mut unique = orders.clone();
    unique.sort_unstable();
    unique.dedup();
    let count = unique.iter().map(|&v| v as u128).sum::<u128>();
    let bytes =
        (count * 2 + vector.len() as u128 * 256) * (u128::from(base + 4096).div_ceil(8) + 96);
    let scratch = orders.iter().copied().max().unwrap_or(0) as u128
        * rayon::current_num_threads().min(n + 1) as u128
        * (u128::from(base + 4096).div_ceil(8) + 96)
        * 16;
    if bytes + scratch > super::source_working_budget()? {
        bail!("u-flow exceeds the declared workspace budget");
    }
    let plan = xc_numerics::hp_runtime::plan_gl_precompute(&unique, p);
    let tables =
        xc_numerics::hp_runtime::map_gl_precompute_planned(&unique, plan, |points, schedule| {
            xc_numerics::quadrature::try_gauss_legendre_nodes_scheduled(
                points,
                p,
                cfg.cache_mode,
                schedule,
            )
            .map(|table| (points, table))
        })
        .into_iter()
        .collect::<std::result::Result<std::collections::HashMap<usize, Table>, _>>()?;
    let exponent = vector
        .iter()
        .filter_map(Float::get_exp)
        .max()
        .map_or(0, i64::from);
    let normalized = vector
        .iter()
        .map(|v| scale_float(v, -exponent, base))
        .collect::<Result<Vec<_>>>()?;
    crate::ccm::certified_roots::boundary::rational_budget(normalized.iter(), normalized.len())?;
    let exact = normalized
        .iter()
        .map(|v| v.to_rational().expect("finite validated source"))
        .collect::<Vec<_>>();
    let events = super::super::try_prime_powers_up_to(params.lambda_sq_int())?;
    for guard in [64, 128, 256, 512, 1024, 2048, 4096] {
        let work = base + guard;
        let length = I::from_float(l, work)?;
        let integrals = (0..=n)
            .into_par_iter()
            .map(|mode| {
                let (nodes, weights) = &tables[&orders[mode]];
                integral(mode, &length, nodes, weights, work)
            })
            .collect::<Result<Vec<_>>>()?;
        let poles = pole(&exact, n, &length, work)?;
        let arches = (0..vector.len())
            .into_par_iter()
            .map(|row| -> Result<I> {
                let (coeff, diag) = arch_coefficients(&exact, n, row);
                let mut terms = coeff
                    .iter()
                    .zip(&integrals)
                    .filter(|(c, _)| *c != &0)
                    .map(|(c, a)| I::from_rational(c, work).mul(&a[0]))
                    .collect::<Vec<_>>();
                let a = &integrals[row.abs_diff(n)];
                terms.push(I::from_rational(&diag, work).mul(&a[1].sub(&a[2])));
                sum(&terms, work)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut primes = vec![I::from_i64(0, work); vector.len()];
        for &(power, prime, _) in &events {
            let action =
                prime_response_kernel::evaluate(n, power, prime, l, &normalized, work, true)?;
            for (value, term) in primes.iter_mut().zip(action.action) {
                *value = value.add(&rounding_enclosure(term)?);
            }
        }
        let mut pole_out = Vec::with_capacity(vector.len());
        let mut arch_out = pole_out.clone();
        let mut prime_out = pole_out.clone();
        let mut total_out = pole_out.clone();
        let mut resolved = true;
        for row in 0..vector.len() {
            let a = &poles[row];
            let b = &arches[row];
            let c = &primes[row];
            let total = a.add(b).add(c);
            let values = [a, b, c, &total]
                .into_iter()
                .map(|v| rounded(&scale(v, exponent)?, p))
                .collect::<Result<Option<Vec<_>>>>()?;
            let Some(values) = values else {
                resolved = false;
                break;
            };
            pole_out.push(values[0].clone());
            arch_out.push(values[1].clone());
            prime_out.push(values[2].clone());
            total_out.push(values[3].clone());
        }
        if resolved {
            return Ok(UFlowVelocityActions {
                tau_pole: pole_out,
                tau_archimedean: arch_out,
                tau_prime: prime_out,
                tau_total: total_out,
            });
        }
    }
    bail!("u-flow action rounding unresolved within 4096 guard bits")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn uflow_directed_quadrature_matches_independent_derivative_rule() {
        for p in [64, 128, 256] {
            let (nodes, weights) = xc_numerics::quadrature::try_gauss_legendre_nodes_scheduled(
                17,
                p,
                xc_numerics::quadrature::CacheMode::default(),
                xc_numerics::hp_runtime::plan_gl_precompute(&[17], p).root_schedule(17),
            )
            .unwrap();
            for n in 0..=3 {
                for length in [1, 3, 7] {
                    let l = Float::with_val(p, length);
                    let got =
                        integral(n, &I::from_float(&l, 2048).unwrap(), &nodes, &weights, 2048)
                            .unwrap();
                    let direct = compute_archimedean_integral_velocities_l(
                        n as i64,
                        &Float::with_val(2048, &l),
                        2048,
                        &nodes
                            .iter()
                            .map(|v| Float::with_val(2048, v))
                            .collect::<Vec<_>>(),
                        &weights
                            .iter()
                            .map(|v| Float::with_val(2048, v))
                            .collect::<Vec<_>>(),
                    );
                    for (a, b) in got.iter().zip([direct.0, direct.1, direct.2]) {
                        assert_eq!(rounded(a, p).unwrap().unwrap(), Float::with_val(p, b));
                    }
                }
            }
        }
    }

    #[test]
    fn uflow_pole_action_matches_independent_quotient_derivative() {
        for p in [64, 128, 256] {
            for n in 0..=3usize {
                for length in [1, 3, 7] {
                    for exponent in [-1000i32, 0, 1000] {
                        let w = 2048;
                        let l = Float::with_val(w, length);
                        let a = Float::with_val(w, rug::float::Constant::Pi).square() * 16;
                        let quarter = (Float::with_val(w, &l) / 4u32).sinh().square();
                        let half = (Float::with_val(w, &l) / 2u32).sinh();
                        let l2 = l.clone().square();
                        let vector = (0..2 * n + 1)
                            .map(|k| Float::with_val(p, k * k + 1) << exponent)
                            .collect::<Vec<_>>();
                        let exact = vector
                            .iter()
                            .map(|v| v.to_rational().unwrap())
                            .collect::<Vec<_>>();
                        let got = pole(&exact, n, &I::from_float(&l, w).unwrap(), w).unwrap();
                        for (row, bound) in got.iter().enumerate() {
                            let m = row as i64 - n as i64;
                            let mut expected = Float::with_val(w, 0);
                            for (col, v) in vector.iter().enumerate() {
                                let k = col as i64 - n as i64;
                                let numerator =
                                    Float::with_val(w, &l2) - Float::with_val(w, &a) * (m * k);
                                let left =
                                    Float::with_val(w, &l2) + Float::with_val(w, &a) * (m * m);
                                let right =
                                    Float::with_val(w, &l2) + Float::with_val(w, &a) * (k * k);
                                let denominator = Float::with_val(w, &left) * &right;
                                let derivative = (Float::with_val(w, &quarter)
                                    + Float::with_val(w, &l) * &half / 4u32)
                                    * &numerator
                                    + Float::with_val(w, &l2) * &quarter * 2u32;
                                let value = derivative * 32u32 / &denominator
                                    - Float::with_val(w, &l)
                                        * &quarter
                                        * &numerator
                                        * 32u32
                                        * (left + right)
                                        * &l
                                        * 2u32
                                        / denominator.square();
                                expected += value * v;
                            }
                            assert_eq!(
                                rounded(bound, p).unwrap().unwrap(),
                                Float::with_val(p, expected),
                                "p={p},n={n},L={length},e={exponent},row={row}"
                            );
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn uflow_exact_odd_and_zero_sources_preserve_zero_center() {
        let p = 128;
        let mut cfg = HighPrecConfig::for_decimal_digits(20);
        cfg.precision_bits = p;
        cfg.quad_points = 32;
        let params = CcmParams::from_lambda_sq_integer(3, 1);
        let l = Float::with_val(p, 3);
        for v in [[1, 0, -1], [0, 0, 0]] {
            let v = v.map(|x| Float::with_val(p, x));
            let action = evaluate(&params, &cfg, &l, &v).unwrap();
            for channel in [
                action.tau_pole,
                action.tau_archimedean,
                action.tau_prime,
                action.tau_total,
            ] {
                assert!(channel[1].is_zero());
                assert_eq!(channel[0], -channel[2].clone());
            }
        }
    }
}
