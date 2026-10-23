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
    // For a nonzero dyadic M*2^(e-b), cancel M's trailing zero bits.
    // Its reduced denominator is 2^(b-e-v2(M)). Adding one preserves that
    // denominator, so multiplication by 2n cannot be integral when the
    // denominator needs more powers of two than 2n supplies. Borrowing the
    // significand allocates nothing; the original rational route handles all
    // divisible cases, zeros, and integers (including its i64 admission).
    if twice_mode != 0 {
        if let (Some(significand), Some(exponent)) = (node.get_significand(), node.get_exp()) {
            if let Some(trailing) = significand.find_one(0) {
                let denominator_bits = i64::from(significand.significant_bits())
                    - i64::from(exponent)
                    - i64::from(trailing);
                if denominator_bits > i64::from(twice_mode.trailing_zeros()) {
                    return Ok(None);
                }
            }
        }
    }
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
fn validate_table(nodes: &[Float], weights: &[Float]) -> Result<()> {
    if nodes.len() != weights.len()
        || nodes.is_empty()
        || nodes.len() > 1_000_000
        || nodes.iter().any(|x| !x.is_finite() || x <= &-1 || x >= &1)
        || weights.iter().any(|x| !x.is_finite() || x <= &0)
    {
        bail!("invalid stored u-flow quadrature rule");
    }
    Ok(())
}
fn integral(n: usize, l: &I, nodes: &[Float], weights: &[Float], p: u32) -> Result<[I; 3]> {
    validate_table(nodes, weights)?;
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
            let (sin, cos) = phase.sin_cos();
            (sin, cos, phase.mul(&half).sin().square().mul(&two).neg())
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
struct IntegrandFactors {
    h: I,
    weight: I,
    combined: I,
    decay_difference: I,
    gamma_offset: I,
}
struct SharedRule {
    factors: Vec<IntegrandFactors>,
    one: I,
    two: I,
    half: I,
    pi: I,
    gamma_tail: I,
    _reservation: xc_core::MemoryReservation,
}
fn shared_rule_bytes(points: usize, p: u32) -> u128 {
    // Five intervals per node, four scalar intervals and the final tail.
    (points as u128 * 10 + 16) * (u128::from(p).div_ceil(8) + 96)
}
fn prepare_shared_rule(l: &I, nodes: &[Float], weights: &[Float], p: u32) -> Result<SharedRule> {
    validate_table(nodes, weights)?;
    let reservation = prime_response_kernel::reserve_optional(shared_rule_bytes(nodes.len(), p))
        .ok_or_else(|| anyhow::anyhow!("optional u-flow rule sharing unavailable"))?;
    let one = I::from_i64(1, p);
    let two = I::from_i64(2, p);
    let half = I::point(Float::with_val(p, 0.5));
    let pi = I::pi(p);
    let factors = nodes
        .iter()
        .zip(weights)
        .map(|(node, weight)| -> Result<_> {
            let h = I::from_float(node, p)?.add(&one).mul(&half);
            let x = l.mul(&h);
            let e = x.mul(&half).neg().exp();
            let rho = e.div(&exp_m1(&x.mul(&two).neg())?.neg())?;
            let coth = one.add(&two.div(&exp_m1(&x.mul(&two))?)?);
            let combined = rho.mul(&one.add(&x.mul(&half.sub(&coth))));
            Ok(IntegrandFactors {
                h,
                weight: I::from_float(weight, p)?,
                combined,
                decay_difference: exp_m1(&x.mul(&half).neg())?,
                gamma_offset: x.mul(&e).mul(&rho).mul(&half),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let gamma_tail = one.div(&two.mul(&sinh(l)?))?;
    Ok(SharedRule {
        factors,
        one,
        two,
        half,
        pi,
        gamma_tail,
        _reservation: reservation,
    })
}
fn integral_shared(n: usize, nodes: &[Float], rule: &SharedRule, p: u32) -> Result<[I; 3]> {
    let SharedRule {
        one, two, half, pi, ..
    } = rule;
    let mut alpha = Vec::with_capacity(nodes.len());
    let mut beta = Vec::with_capacity(nodes.len());
    let mut gamma = Vec::with_capacity(nodes.len());
    for (node, factor) in nodes.iter().zip(&rule.factors) {
        let phase = pi.mul(&I::from_u64((2 * n) as u64, p)).mul(&factor.h);
        let quadrant = exact_quadrant(node, n)?;
        let (sin, cos, difference_cos) = if let Some(q) = quadrant {
            let cos = I::from_i64([1, 0, -1, 0][q], p);
            (I::from_i64([0, 1, 0, -1][q], p), cos.clone(), cos.sub(one))
        } else {
            let (sin, cos) = phase.sin_cos();
            (sin, cos, phase.mul(half).sin().square().mul(two).neg())
        };
        alpha.push(factor.weight.mul(&sin).mul(&factor.combined));
        beta.push(factor.weight.mul(&factor.h).mul(&cos).mul(&factor.combined));
        let difference = difference_cos.sub(&factor.decay_difference);
        gamma.push(
            factor
                .weight
                .mul(&difference.mul(&factor.combined).add(&factor.gamma_offset)),
        );
    }
    Ok([
        sum(&alpha, p)?.div(&two.mul(pi))?,
        sum(&beta, p)?.mul(half),
        sum(&gamma, p)?.mul(half).add(&rule.gamma_tail),
    ])
}
#[cfg(test)]
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
    evaluate_with_reuse(params, cfg, l, vector, true)
}
#[cfg(test)]
thread_local! {
    static GL_PRECOMPUTE_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}
fn evaluate_with_reuse(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    l: &Float,
    vector: &[Float],
    reuse: bool,
) -> Result<UFlowVelocityActions> {
    let event_batch = if reuse {
        prime_response_kernel::validate_batch_configuration()?;
        match std::env::var("XC_RESPONSE_EVENT_BATCH") {
            Ok(value) => event_batch_size(Some(&value))?,
            Err(std::env::VarError::NotPresent) => event_batch_size(None)?,
            Err(_) => bail!("XC_RESPONSE_EVENT_BATCH must be valid text"),
        }
    } else {
        1
    };
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
    let budget = super::source_working_budget()?;
    if bytes + scratch > budget {
        bail!("u-flow exceeds the declared workspace budget");
    }
    // Refuse an inadmissible exact source before resolving potentially expensive
    // quadrature tables. Successful inputs retain the same source arithmetic.
    let exponent = vector
        .iter()
        .filter_map(Float::get_exp)
        .max()
        .map_or(0, i64::from);
    let normalized = vector
        .iter()
        .map(|v| scale_float(v, -exponent, base))
        .collect::<Result<Vec<_>>>()?;
    let exact = crate::ccm::certified_roots::boundary::response_points(
        &normalized,
        n,
        rayon::current_num_threads(),
        bytes + scratch,
        budget,
    )?;
    #[cfg(test)]
    GL_PRECOMPUTE_CALLS.with(|calls| calls.set(calls.get() + 1));
    let tables = {
        let plan = xc_numerics::hp_runtime::plan_gl_precompute(&unique, p);
        let _stage = crate::ccm::capture_runtime::Stage::new(format!(
            "u-flow quadrature tables ({} unique, {}..{} points, {p} bits, {})",
            unique.len(),
            unique.first().copied().unwrap_or(0),
            unique.last().copied().unwrap_or(0),
            plan.label(),
        ));
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
        .collect::<std::result::Result<std::collections::HashMap<usize, Table>, _>>()?
    };
    let events = super::super::try_prime_powers_up_to(params.lambda_sq_int())?;
    // Exact preparation is independent of the guard precision and prime event.
    // Keep the previous path if the extra preparation cannot be admitted.
    let prime_scratch = prime_response_kernel::workspace_bytes(vector.len(), base + 4096, true);
    let coefficient_source = prime_response_kernel::CoefficientSource::new(
        &exact.values,
        n,
        if reuse {
            budget.saturating_sub(bytes + scratch + prime_scratch + exact.workspace_bytes)
        } else {
            0
        },
    );
    // The order formula has one repeated prefix, then strictly increasing
    // orders. Sharing only that prefix bounds storage to one rule per guard.
    let shared_order = orders
        .first()
        .copied()
        .filter(|order| reuse && orders.iter().filter(|v| *v == order).count() > 1);
    for guard in [64, 128, 256, 512, 1024, 2048, 4096] {
        let work = base + guard;
        let length = I::from_float(l, work)?;
        let integral_stage = crate::ccm::capture_runtime::Stage::new(format!(
            "u-flow integrals (guard={guard}, work={work} bits, {} modes)",
            n + 1,
        ));
        let shared = shared_order.and_then(|order| {
            let required = shared_rule_bytes(order, work);
            if required > (64u128 << 20)
                || required
                    > budget.saturating_sub(bytes + scratch + coefficient_source.storage_bytes())
            {
                return None;
            }
            let (nodes, weights) = &tables[&order];
            // A preparation error follows the original mode/node path so its
            // existing failure order is preserved rather than reordered.
            prepare_shared_rule(&length, nodes, weights, work).ok()
        });
        let integrals = (0..=n)
            .into_par_iter()
            .map(|mode| {
                let (nodes, weights) = &tables[&orders[mode]];
                if shared_order == Some(orders[mode]) {
                    if let Some(rule) = &shared {
                        return integral_shared(mode, nodes, rule, work);
                    }
                }
                integral(mode, &length, nodes, weights, work)
            })
            .collect::<Result<Vec<_>>>()?;
        drop(shared);
        drop(integral_stage);
        let assembly_stage = crate::ccm::capture_runtime::Stage::new(format!(
            "u-flow pole and archimedean actions (guard={guard}, work={work} bits)"
        ));
        let poles = pole(&exact.values, n, &length, work)?;
        let arches = (0..vector.len())
            .into_par_iter()
            .map(|row| -> Result<I> {
                let (coeff, diag) = coefficient_source.arch_coefficients(row);
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
        drop(assembly_stage);
        let prime_stage = crate::ccm::capture_runtime::Stage::new(format!(
            "u-flow prime actions (guard={guard}, work={work} bits, {} events)",
            events.len(),
        ));
        let mut primes = vec![I::from_i64(0, work); vector.len()];
        for batch in events.chunks(event_batch) {
            let actions = if reuse {
                let pairs = batch
                    .iter()
                    .map(|&(power, prime, _)| (power, prime))
                    .collect::<Vec<_>>();
                prime_response_kernel::evaluate_prepared_batch(
                    n,
                    &pairs,
                    l,
                    &normalized,
                    work,
                    true,
                    &coefficient_source,
                    budget.saturating_sub(
                        bytes
                            + scratch
                            + prime_scratch
                            + exact.workspace_bytes
                            + coefficient_source.storage_bytes(),
                    ),
                )?
            } else {
                None
            };
            // Retain the batch owner (and its memory reservation) until every
            // independently rounded event has been accumulated in canonical order.
            if let Some(actions) = actions {
                for action in &actions.actions {
                    for (value, term) in primes.iter_mut().zip(&action.action) {
                        *value = value.add(&rounding_enclosure(term.clone())?);
                    }
                }
            } else {
                // Optional batching must never add a source-admission failure.
                // A denied batch uses the original single-event storage lifetime.
                for &(power, prime, _) in batch {
                    let action = prime_response_kernel::evaluate_prepared(
                        n,
                        power,
                        prime,
                        l,
                        &normalized,
                        work,
                        true,
                        &coefficient_source,
                    )?;
                    for (value, term) in primes.iter_mut().zip(action.action) {
                        *value = value.add(&rounding_enclosure(term)?);
                    }
                }
            }
        }
        drop(prime_stage);
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

fn event_batch_size(value: Option<&str>) -> Result<usize> {
    match value {
        None => Ok(8),
        Some(value) => {
            anyhow::ensure!(
                !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()),
                "XC_RESPONSE_EVENT_BATCH must be a positive ASCII integer"
            );
            let size: usize = value.parse().map_err(|_| {
                anyhow::anyhow!("XC_RESPONSE_EVENT_BATCH must be a positive integer")
            })?;
            anyhow::ensure!(size > 0, "XC_RESPONSE_EVENT_BATCH must be positive");
            Ok(size)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn response_event_batch_configuration_is_positive_and_not_a_source_cap() {
        assert_eq!(event_batch_size(None).unwrap(), 8);
        assert_eq!(event_batch_size(Some("1")).unwrap(), 1);
        assert_eq!(event_batch_size(Some("256")).unwrap(), 256);
        for value in ["0", "-1", "+1", " 1", "１", "", "many"] {
            assert!(event_batch_size(Some(value)).is_err());
        }
    }
    #[test]
    fn capacity_uflow_denied_exact_source_skips_quadrature() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        pool.install(|| {
            let resources = xc_core::RunResources::with_worker_limit(4 << 20, 2).unwrap();
            let _scope = resources.enter();
            let mut cfg = HighPrecConfig::for_decimal_digits(20);
            cfg.precision_bits = 128;
            cfg.quad_points = 17;
            cfg.cache_mode = xc_numerics::quadrature::CacheMode::Off;
            let params = CcmParams::from_lambda_sq_integer(3, 1);
            let vector = [
                Float::with_val(128, 1),
                Float::with_val(128, 0),
                Float::with_val(128, 1) >> 1_000_000u32,
            ];
            let before = GL_PRECOMPUTE_CALLS.with(std::cell::Cell::get);
            let error = evaluate(&params, &cfg, &Float::with_val(128, 3), &vector)
                .err()
                .expect("the compact Float source exceeds exact rational workspace");
            assert!(error
                .to_string()
                .contains("exact response point conversion exceeds the rational workspace budget"));
            assert_eq!(GL_PRECOMPUTE_CALLS.with(std::cell::Cell::get), before);
            assert_eq!(resources.usage().working_bytes, 0);
            assert_eq!(resources.usage().retained_bytes, 0);
        });
    }

    #[test]
    fn uflow_paired_trigonometry_matches_separate_directed_endpoints() {
        for p in [64, 128, 6772, 13608] {
            let pi = I::pi(p);
            let phases = [
                I::from_i64(0, p),
                I::point(-Float::with_val(p, 0)),
                I::point(Float::with_val(p, 1) >> 400u32),
                pi.mul(&I::from_i64(800, p)),
                pi.div(&I::from_i64(7, p)).unwrap(),
                I::new(Float::with_val(p, -100), Float::with_val(p, 100)).unwrap(),
            ];
            for phase in phases {
                let (sin, cos) = phase.sin_cos();
                for (actual, expected) in [(sin, phase.sin()), (cos, phase.cos())] {
                    for (a, b) in [
                        (actual.lower(), expected.lower()),
                        (actual.upper(), expected.upper()),
                    ] {
                        assert_eq!(a, b);
                        assert_eq!(a.prec(), b.prec());
                        assert_eq!(a.is_sign_negative(), b.is_sign_negative());
                    }
                }
            }
        }
    }
    #[test]
    #[ignore = "supervised shared-integrand measurement; not claim-scale qualification"]
    fn measure_uflow_shared_rule_kernel() {
        use std::{hint::black_box, time::Instant};
        let resources = xc_core::RunResources::new(1 << 28).unwrap();
        let _scope = resources.enter();
        let (modes, points, p) = (16, 65, 512);
        let (nodes, weights) = xc_numerics::quadrature::try_gauss_legendre_nodes_scheduled(
            points,
            p,
            xc_numerics::quadrature::CacheMode::Off,
            xc_numerics::hp_runtime::plan_gl_precompute(&[points], p).root_schedule(points),
        )
        .unwrap();
        let l = I::from_i64(3, p);
        let scalar_bits = |values: &[I; 3]| {
            values
                .iter()
                .flat_map(|v| [v.lower(), v.upper()])
                .map(|v| {
                    (
                        xc_numerics::prefix::lossless_decimal(v),
                        v.prec(),
                        v.is_sign_negative(),
                    )
                })
                .collect::<Vec<_>>()
        };
        let shared = prepare_shared_rule(&l, &nodes, &weights, p).unwrap();
        for mode in 0..=modes {
            assert_eq!(
                scalar_bits(&integral(mode, &l, &nodes, &weights, p).unwrap()),
                scalar_bits(&integral_shared(mode, &nodes, &shared, p).unwrap())
            );
        }
        drop(shared);
        let mut times = [Vec::new(), Vec::new()];
        for repetition in 0..5 {
            for route in if repetition % 2 == 0 { [0, 1] } else { [1, 0] } {
                let started = Instant::now();
                let shared =
                    (route == 1).then(|| prepare_shared_rule(&l, &nodes, &weights, p).unwrap());
                for mode in 0..=modes {
                    black_box(
                        if let Some(shared) = &shared {
                            integral_shared(mode, &nodes, shared, p)
                        } else {
                            integral(mode, &l, &nodes, &weights, p)
                        }
                        .unwrap(),
                    );
                }
                times[route].push(started.elapsed().as_secs_f64());
            }
        }
        for values in &mut times {
            values.sort_by(f64::total_cmp);
        }
        println!("Directed integral kernel only: modes={modes}, points={points}, p={p}, trials=5; original/current median {:.6}/{:.6}s; shared preparation included, table generation excluded; exact endpoint equality checked; not a claim-scale speed estimate", times[0][2], times[1][2]);
    }
    fn channel_bits(action: &UFlowVelocityActions) -> Vec<(String, u32, bool)> {
        [
            &action.tau_pole,
            &action.tau_archimedean,
            &action.tau_prime,
            &action.tau_total,
        ]
        .into_iter()
        .flatten()
        .map(|v| {
            (
                xc_numerics::prefix::lossless_decimal(v),
                v.prec(),
                v.is_sign_negative(),
            )
        })
        .collect()
    }
    #[test]
    fn capacity_uflow_full_actions_match_original_with_admitted_exact_source() {
        let resources = xc_core::RunResources::new(1 << 30).unwrap();
        let _scope = resources.enter();
        for p in [64, 128] {
            let mut cfg = HighPrecConfig::for_decimal_digits(20);
            cfg.precision_bits = p;
            cfg.quad_points = 96;
            cfg.cache_mode = xc_numerics::quadrature::CacheMode::Off;
            let params = CcmParams::from_lambda_sq_integer(3, 1);
            let l = Float::with_val(p, 3);
            for values in [[2, -3, 1], [1, 0, -1], [0, 0, 0]] {
                let vector = values.map(|v| Float::with_val(p, v));
                let original = evaluate_with_reuse(&params, &cfg, &l, &vector, false).unwrap();
                let reused = evaluate(&params, &cfg, &l, &vector).unwrap();
                assert_eq!(channel_bits(&reused), channel_bits(&original));
                assert_eq!(resources.usage().retained_bytes, 0);
                assert_eq!(resources.usage().working_bytes, 0);
                assert!(resources.usage().peak_bytes > 0);
                let occupied = resources.try_reserve_retained((1 << 30) - 1).unwrap();
                let before = resources.usage();
                let error = evaluate(&params, &cfg, &l, &vector).err().unwrap();
                assert!(format!("{error:#}").contains("run memory budget exceeded"));
                assert_eq!(resources.usage(), before);
                drop(occupied);
            }
        }
    }

    #[test]
    fn batched_uflow_preserves_all_channels_across_multiple_prime_events() {
        let resources = xc_core::RunResources::new(1 << 30).unwrap();
        let _scope = resources.enter();
        let p = 128;
        let mut cfg = HighPrecConfig::for_decimal_digits(20);
        cfg.precision_bits = p;
        cfg.quad_points = 96;
        cfg.cache_mode = xc_numerics::quadrature::CacheMode::Off;
        let params = CcmParams::from_lambda_sq_integer(13, 3);
        let l = Float::with_val(p, 13).ln();
        let vector = (0..7)
            .map(|k| Float::with_val(p, k - 2) / (k + 3))
            .collect::<Vec<_>>();
        let original = evaluate_with_reuse(&params, &cfg, &l, &vector, false).unwrap();
        let batched = evaluate_with_reuse(&params, &cfg, &l, &vector, true).unwrap();
        assert_eq!(channel_bits(&original), channel_bits(&batched));
        assert_eq!(resources.usage().working_bytes, 0);
        assert_eq!(resources.usage().retained_bytes, 0);
    }
    #[test]
    fn uflow_shared_rule_reservation_denial_releases_all_optional_storage() {
        let resources = xc_core::RunResources::new(1 << 20).unwrap();
        let _scope = resources.enter();
        let p = 128;
        let l = I::from_i64(3, p);
        let nodes = [-0.5, 0.0, 0.5].map(|v| Float::with_val(p, v));
        let weights = [1, 2, 1].map(|v| Float::with_val(p, v));
        {
            let _rule = prepare_shared_rule(&l, &nodes, &weights, p).unwrap();
            assert_eq!(
                u128::from(resources.usage().retained_bytes),
                shared_rule_bytes(nodes.len(), p)
            );
        }
        assert_eq!(resources.usage().retained_bytes, 0);
        let _occupied = resources.try_reserve_retained((1 << 20) - 1).unwrap();
        assert!(prepare_shared_rule(&l, &nodes, &weights, p).is_err());
        assert_eq!(resources.usage().retained_bytes, (1 << 20) - 1);
    }
    #[test]
    fn uflow_shared_rule_matches_original_interval_endpoints_across_guards() {
        let resources = xc_core::RunResources::new(1 << 30).unwrap();
        let _scope = resources.enter();
        for source_precision in [64, 128] {
            let mut nodes = [-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75]
                .map(|x| Float::with_val(source_precision, x))
                .to_vec();
            nodes.push(Float::with_val(source_precision, 1) >> 400u32);
            nodes.push(-Float::with_val(source_precision, 1) >> 400u32);
            let weights = (0..nodes.len())
                .map(|i| Float::with_val(source_precision, i + 1) / 16u32)
                .collect::<Vec<_>>();
            for guard in [64, 128, 256] {
                let work = source_precision + guard;
                for length in [1, 3] {
                    let l = I::from_i64(length, work);
                    let shared = prepare_shared_rule(&l, &nodes, &weights, work).unwrap();
                    for mode in 0..=4 {
                        let expected = integral(mode, &l, &nodes, &weights, work).unwrap();
                        let actual = integral_shared(mode, &nodes, &shared, work).unwrap();
                        for (actual, expected) in actual.iter().zip(&expected) {
                            for (a, b) in [
                                (actual.lower(), expected.lower()),
                                (actual.upper(), expected.upper()),
                            ] {
                                assert_eq!(a, b, "p={source_precision},guard={guard},mode={mode}");
                                assert_eq!(a.prec(), b.prec());
                                assert_eq!(a.is_sign_negative(), b.is_sign_negative());
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn uflow_compact_arch_coefficients_match_original_exact_rows() {
        let resources = xc_core::RunResources::new(1 << 30).unwrap();
        let _scope = resources.enter();
        for modes in 0..=8 {
            let exact = (0..2 * modes + 1)
                .map(|k| Rational::from(((k as i32 - 5) * (k as i32 + 1), 1u32 << k)))
                .collect::<Vec<_>>();
            let prepared =
                prime_response_kernel::CoefficientSource::new(&exact, modes, 1u128 << 30);
            let fallback = prime_response_kernel::CoefficientSource::new(&exact, modes, 0);
            assert!(prepared.storage_bytes() > 0);
            assert_eq!(fallback.storage_bytes(), 0);
            for row in 0..exact.len() {
                let expected = arch_coefficients(&exact, modes, row);
                assert_eq!(prepared.arch_coefficients(row), expected);
                assert_eq!(fallback.arch_coefficients(row), expected);
            }
        }
    }

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

    fn exact_quadrant_rational_reference(node: &Float, n: usize) -> Result<Option<usize>> {
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

    fn compare_quadrant_routes(node: &Float, mode: usize) {
        let render = |value: Result<Option<usize>>| value.map_err(|error| format!("{error:#}"));
        assert_eq!(
            render(exact_quadrant(node, mode)),
            render(exact_quadrant_rational_reference(node, mode)),
            "precision={}, exponent={:?}, negative={}, mode={mode}",
            node.prec(),
            node.get_exp(),
            node.is_sign_negative()
        );
    }

    #[test]
    fn uflow_quadrant_fast_rejection_matches_exact_dyadics() {
        let modes = [0, 1, 2, 3, 4, 7, 8, 15, 16, 63, 64, 255, 256, 800, 4096];
        for precision in [16, 64, 128, 6772] {
            for numerator in -17..=17 {
                for shift in -12..=2 {
                    let node = Float::with_val(precision, numerator) << shift;
                    for mode in modes {
                        compare_quadrant_routes(&node, mode);
                    }
                }
            }
            // Full-precision dyadics obtained from non-dyadic divisions,
            // with deterministic varied signs and exponents.
            let mut state = 0x79a5_d13c_e012_638fu64;
            for index in 0..64 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let mut node =
                    Float::with_val(precision, state) / Float::with_val(precision, u64::MAX - 58);
                node <<= [-400, -64, 0, 5, 32][index % 5];
                if index % 2 == 0 {
                    node = -node;
                }
                for mode in modes {
                    compare_quadrant_routes(&node, mode);
                }
            }
        }
    }

    #[test]
    fn uflow_quadrant_fast_rejection_preserves_edge_cases_and_errors() {
        let precision = 128;
        let tiny = Float::with_val(precision, 1) >> 4096u32;
        for node in [
            Float::with_val(precision, 0),
            -Float::with_val(precision, 0),
            Float::with_val(precision, -1),
            Float::with_val(precision, 1),
            Float::with_val(precision, -0.5),
            Float::with_val(precision, 0.5),
            tiny.clone(),
            -tiny,
        ] {
            for mode in [0, 1, 2, 4096, usize::MAX / 2, usize::MAX] {
                compare_quadrant_routes(&node, mode);
            }
        }
        // The original rational resource check still wins before mode overflow,
        // and before the new rejection can bypass an expensive conversion.
        let denied = Float::with_val(precision, 1) >> 70_000_000u32;
        for mode in [0, 1, usize::MAX] {
            compare_quadrant_routes(&denied, mode);
            assert!(format!("{:#}", exact_quadrant(&denied, mode).unwrap_err())
                .contains("rational workspace budget"));
        }
        for special in [rug::float::Special::Nan, rug::float::Special::Infinity] {
            compare_quadrant_routes(&Float::with_val(precision, special), usize::MAX);
        }
    }

    #[test]
    #[ignore = "bounded kernel benchmark; no quadrature generation or cache access"]
    fn uflow_quadrant_fast_rejection_benchmark() {
        let nodes = (-64..=64)
            .filter(|value| *value != 0)
            .map(|value| Float::with_val(6772, value) / 257u32)
            .collect::<Vec<_>>();
        for node in &nodes {
            for mode in 0..=32 {
                compare_quadrant_routes(node, mode);
            }
        }
        let mut timings = [Vec::new(), Vec::new()];
        for _ in 0..5 {
            for (route, times) in timings.iter_mut().enumerate() {
                let started = std::time::Instant::now();
                for mode in 1..=32 {
                    for node in &nodes {
                        let value = if route == 0 {
                            exact_quadrant_rational_reference(node, mode)
                        } else {
                            exact_quadrant(node, mode)
                        };
                        std::hint::black_box(value.unwrap());
                    }
                }
                times.push(started.elapsed().as_secs_f64());
            }
        }
        for times in &mut timings {
            times.sort_by(f64::total_cmp);
        }
        println!("Exact quadrant kernel: p=6772, 128 stored GL-like dyadics, 32 modes, 5 trials; rational/borrowed-rejection median {:.6}/{:.6}s; source construction excluded; exact Options/errors compared; not a full u-flow speed estimate", timings[0][2], timings[1][2]);
    }
}
