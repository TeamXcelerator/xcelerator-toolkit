//! Correctly rounded finite matrix stages. Quadrature tables and previously
//! rounded integral/component values are explicit stored point inputs; this
//! does not supply a continuum or quadrature truncation bound.
use super::*;
use rug::{
    float::{Constant, Round},
    Rational,
};
use xc_numerics::mpfr_interval::MpfrInterval as I;
#[cfg(test)]
#[path = "matrix_budget.rs"]
mod matrix_budget;

// Dense retained/output matrices have p-bit entries. Guard precision belongs
// only to scalar and linear-size interval workspaces, not every dense entry.
fn matrix_workspace_bytes(d: usize, base: u32) -> u128 {
    let stored = u128::from(base).div_ceil(8) + 96;
    let guarded = u128::from(base * 2 + 4096).div_ceil(8) + 96;
    4 * d as u128 * d as u128 * stored + (32 * d as u128 + 512) * guarded
}
pub(super) fn preflight(n: usize, l: &Float, p: u32) -> Result<()> {
    dimensions(n, l, p).map(|_| ())
}

fn rational_stage_budget<'a>(
    dimension: usize,
    base: u32,
    points: impl Iterator<Item = &'a Float>,
    parallelism: usize,
) -> Result<()> {
    let matrix_bytes = matrix_workspace_bytes(dimension, base);
    // Dyadic products double the bit span; a row sum uses a common power-of-two
    // denominator and at most log2(dimension) extra numerator bits. The point
    // workspace's 64 maximum-sized temporaries cover the scalar/row operations.
    // Charge every concurrently active row and the existing matrix workspace.
    let rational_bytes =
        crate::ccm::certified_roots::boundary::rational_point_vector_workspace(points)?;
    if matrix_bytes + rational_bytes * parallelism.max(1) as u128 > (8u128 << 30) {
        bail!("matrix point and rational stages exceed the 8 GiB workspace budget");
    }
    Ok(())
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
fn sinh(x: &I) -> Result<I> {
    x.validate()?;
    let mut lo = x.lower().clone();
    let mut hi = x.upper().clone();
    lo.sinh_round(Round::Down);
    hi.sinh_round(Round::Up);
    Ok(I::new(lo, hi)?)
}
fn exp_m1(x: &I) -> Result<I> {
    x.validate()?;
    let mut lo = x.lower().clone();
    let mut hi = x.upper().clone();
    lo.exp_m1_round(Round::Down);
    hi.exp_m1_round(Round::Up);
    Ok(I::new(lo, hi)?)
}
fn sum(x: &[I], p: u32) -> Result<I> {
    for v in x {
        v.validate()?;
    }
    Ok(I::new(
        Float::with_val_round(p, Float::sum(x.iter().map(I::lower)), Round::Down).0,
        Float::with_val_round(p, Float::sum(x.iter().map(I::upper)), Round::Up).0,
    )?)
}
fn dimensions(n: usize, l: &Float, p: u32) -> Result<(usize, u32)> {
    if !(64..=1_000_000).contains(&p)
        || n > 4096
        || !l.is_finite()
        || l <= &0
        || l.prec() > 1_000_000
    {
        bail!("invalid matrix point source or supported precision/dimension");
    }
    let d = 2 * n + 1;
    let base = p.max(l.prec());
    if matrix_workspace_bytes(d, base) > 8u128 << 30 {
        bail!("matrix point stage exceeds the 8 GiB workspace budget");
    }
    Ok((d, base))
}
fn rational_round(x: Rational, p: u32) -> Result<Float> {
    let out = Float::with_val(p, &x);
    if !out.is_finite()
        || (out.is_zero() && x != 0)
        || (out.get_exp() == Some(rug::float::exp_min())
            && Float::with_val_round(p, &x, Round::Zero).0.is_zero())
    {
        bail!("matrix point value is outside the supported exponent range");
    }
    Ok(out)
}
fn kappa(l: &I, p: u32) -> Result<I> {
    let half = l.div(&I::from_i64(2, p))?;
    let mut lo = half.lower().clone();
    let mut hi = half.upper().clone();
    lo.tanh_round(Round::Down);
    hi.tanh_round(Round::Up);
    let t = I::new(lo, hi)?;
    let e = I::new(
        Float::with_val_round(p, Constant::Euler, Round::Down).0,
        Float::with_val_round(p, Constant::Euler, Round::Up).0,
    )?;
    Ok(t.mul(&I::pi(p))
        .mul(&I::from_i64(4, p))
        .ln()?
        .add(&e)
        .div(&I::from_i64(2, p))?)
}
/// Guard schedule shared by every archimedean integral evaluation.
const INTEGRAL_GUARDS: [u32; 7] = [64, 128, 256, 512, 1024, 2048, 4096];

/// Mode-independent terms of one quadrature rule at one working precision.
/// Modes that share a rule share these terms; each is formed by exactly the
/// interval operations a single-mode evaluation performs, so a shared table
/// yields bit-identical enclosures and correctly rounded results.
pub(super) struct IntegralNodeTable {
    guard: u32,
    work: u32,
    length: I,
    two: I,
    half: I,
    pi: I,
    kappa: I,
    terms: Vec<IntegralNodeTerm>,
}

struct IntegralNodeTerm {
    /// Exact node plus one; the phase turn is `(node + 1) * 2n`.
    shifted: Rational,
    h: I,
    /// Weight times the archimedean density rho(x).
    weighted_rho: I,
    /// `weighted_rho * x`.
    weighted_rho_x: I,
    /// `expm1(-x/2)`.
    decay: I,
}

fn validate_integral_rule(l: &Float, p: u32, nodes: &[Float], weights: &[Float]) -> Result<u32> {
    let (_, mut base) = dimensions(0, l, p)?;
    if nodes.is_empty()
        || nodes.len() != weights.len()
        || nodes.len() > 1_000_000
        || nodes
            .iter()
            .any(|v| !v.is_finite() || v <= &-1 || v >= &1 || v.prec() > 1_000_000)
        || weights
            .iter()
            .any(|v| !v.is_finite() || v <= &0 || v.prec() > 1_000_000)
    {
        bail!("invalid stored matrix quadrature rule");
    }
    base = base.max(nodes.iter().chain(weights).map(Float::prec).max().unwrap());
    let floating_bytes = nodes.len() as u128 * (u128::from(2 * base + 4096).div_ceil(8) + 96) * 16;
    let rational_bytes =
        crate::ccm::certified_roots::boundary::rational_point_vector_workspace(nodes.iter())?;
    if floating_bytes + rational_bytes > 8u128 << 30 {
        bail!("matrix quadrature exceeds workspace budget");
    }
    crate::ccm::certified_roots::boundary::rational_point_vector_budget(nodes.iter())?;
    Ok(base)
}

/// Prepare the mode-independent terms of a quadrature rule at one guard.
pub(super) fn integral_node_table(
    l: &Float,
    p: u32,
    nodes: &[Float],
    weights: &[Float],
    guard: u32,
) -> Result<IntegralNodeTable> {
    let base = validate_integral_rule(l, p, nodes, weights)?;
    let work = base * 2 + guard;
    let length = I::from_float(l, work)?;
    let one = I::from_i64(1, work);
    let two = I::from_i64(2, work);
    let half = I::point(Float::with_val(work, 0.5));
    let pi = I::pi(work);
    let kappa = kappa(&length, work)?;
    let mut terms = Vec::with_capacity(nodes.len());
    for (node, weight) in nodes.iter().zip(weights) {
        let h = I::from_float(node, work)?.add(&one).mul(&half);
        let x = length.mul(&h);
        let minus_half = x.mul(&half).neg();
        let rho = minus_half.exp().div(&exp_m1(&x.mul(&two).neg())?.neg())?;
        let weighted_rho = I::from_float(weight, work)?.mul(&rho);
        terms.push(IntegralNodeTerm {
            shifted: node.to_rational().unwrap() + 1i32,
            weighted_rho_x: weighted_rho.mul(&x),
            decay: exp_m1(&minus_half)?,
            weighted_rho,
            h,
        });
    }
    Ok(IntegralNodeTable {
        guard,
        work,
        length,
        two,
        half,
        pi,
        kappa,
        terms,
    })
}

/// One guard attempt for mode `n`; `None` when the rounding is unresolved.
fn mode_integrals(
    n: i64,
    p: u32,
    table: &IntegralNodeTable,
) -> Result<Option<(Float, Float, Float)>> {
    let work = table.work;
    let one = I::from_i64(1, work);
    let (two, half, pi) = (&table.two, &table.half, &table.pi);
    let mut aa = Vec::with_capacity(table.terms.len());
    let mut bb = Vec::with_capacity(table.terms.len());
    let mut gg = Vec::with_capacity(table.terms.len());
    for term in &table.terms {
        let phase = pi.mul(&I::from_i64(2 * n, work)).mul(&term.h);
        let turn: Rational = term.shifted.clone() * (2 * n);
        let quadrant = if turn.denom() == &1 {
            let q = turn.numer().to_i64().expect("bounded node and mode");
            Some(q.rem_euclid(4) as usize)
        } else {
            None
        };
        let (sin, cos, cm1) = if let Some(q) = quadrant {
            let cos = I::from_i64([1, 0, -1, 0][q], work);
            (
                I::from_i64([0, 1, 0, -1][q], work),
                cos.clone(),
                cos.sub(&one),
            )
        } else {
            (
                phase.sin(),
                phase.cos(),
                phase.mul(half).sin().square().mul(two).neg(),
            )
        };
        aa.push(term.weighted_rho.mul(&sin));
        bb.push(term.weighted_rho_x.mul(&cos));
        gg.push(term.weighted_rho.mul(&cm1.sub(&term.decay)));
    }
    let a = sum(&aa, work)?.mul(&table.length).div(&two.mul(pi))?;
    let b = sum(&bb, work)?.mul(half);
    let g = sum(&gg, work)?
        .mul(&table.length)
        .mul(half)
        .add(&table.kappa);
    Ok([a, b, g]
        .iter()
        .map(|v| rounded(v, p))
        .collect::<Result<Option<Vec<_>>>>()?
        .map(|v| (v[0].clone(), v[1].clone(), v[2].clone())))
}

pub(super) fn integrals(
    n: i64,
    l: &Float,
    p: u32,
    nodes: &[Float],
    weights: &[Float],
) -> Result<(Float, Float, Float)> {
    dimensions(n.unsigned_abs() as usize, l, p)?;
    integrals_from_guards(n, l, p, nodes, weights, &INTEGRAL_GUARDS)
}

fn integrals_from_guards(
    n: i64,
    l: &Float,
    p: u32,
    nodes: &[Float],
    weights: &[Float],
    guards: &[u32],
) -> Result<(Float, Float, Float)> {
    for &guard in guards {
        let table = integral_node_table(l, p, nodes, weights, guard)?;
        if let Some(values) = mode_integrals(n, p, &table)? {
            return Ok(values);
        }
    }
    bail!("matrix integral rounding unresolved within the guard budget")
}

/// Evaluate mode `n` with a first-guard table shared across modes, escalating
/// through the ordinary guard schedule only when that guard is unresolved.
pub(super) fn integrals_with_table(
    n: i64,
    l: &Float,
    p: u32,
    nodes: &[Float],
    weights: &[Float],
    first: &IntegralNodeTable,
) -> Result<(Float, Float, Float)> {
    dimensions(n.unsigned_abs() as usize, l, p)?;
    if first.guard == INTEGRAL_GUARDS[0] && first.terms.len() == nodes.len() {
        if let Some(values) = mode_integrals(n, p, first)? {
            return Ok(values);
        }
        // The first guard is deterministic and already unresolved.
        return integrals_from_guards(n, l, p, nodes, weights, &INTEGRAL_GUARDS[1..]);
    }
    integrals(n, l, p, nodes, weights)
}
pub(super) fn pole_arch(
    n: usize,
    l: &Float,
    p: u32,
    t: &ComputedArchimedeanIntegrals,
) -> Result<(Vec<Float>, Vec<Float>)> {
    let (d, base) = dimensions(n, l, p)?;
    if [&t.alpha, &t.beta, &t.gamma]
        .iter()
        .any(|v| v.len() != n + 1 || v.iter().any(|x| !x.is_finite() || x.prec() > 1_000_000))
    {
        bail!("invalid matrix integral table shape or precision");
    }
    rational_stage_budget(d, base, t.alpha.iter().chain(&t.beta).chain(&t.gamma), 1)?;
    let alpha = t
        .alpha
        .iter()
        .map(|x| x.to_rational().unwrap())
        .collect::<Vec<_>>();
    let beta = t
        .beta
        .iter()
        .map(|x| x.to_rational().unwrap())
        .collect::<Vec<_>>();
    let gamma = t
        .gamma
        .iter()
        .map(|x| x.to_rational().unwrap())
        .collect::<Vec<_>>();
    let signed = |m: i64| {
        let x = alpha[m.unsigned_abs() as usize].clone();
        if m < 0 {
            -x
        } else {
            x
        }
    };
    let mut arch = vec![Float::with_val(p, 0); d * d];
    for row in 0..d {
        let a = row as i64 - n as i64;
        for col in row..d {
            let b = col as i64 - n as i64;
            let v = if a == b {
                (gamma[a.unsigned_abs() as usize].clone() - &beta[a.unsigned_abs() as usize]) * 2
            } else {
                (signed(b) - signed(a)) / (a - b)
            };
            let v = rational_round(v, p)?;
            arch[row * d + col] = v.clone();
            arch[col * d + row] = v;
        }
    }
    for guard in [64, 128, 256, 512, 1024, 2048, 4096] {
        let work = 2 * base + guard;
        let length = I::from_float(l, work)?;
        let l2 = length.square();
        let c = I::pi(work).square().mul(&I::from_i64(16, work));
        let prefactor = sinh(&length.div(&I::from_i64(4, work))?)?
            .square()
            .mul(&length)
            .mul(&I::from_i64(32, work));
        let den = (0..=n)
            .map(|k| l2.add(&c.mul(&I::from_u64((k * k) as u64, work))))
            .collect::<Vec<_>>();
        let rows = (0..d)
            .into_par_iter()
            .map(|row| -> Result<Option<Vec<Float>>> {
                let a = row as i64 - n as i64;
                (0..d)
                    .map(|col| {
                        let b = col as i64 - n as i64;
                        let numerator = l2.sub(&c.mul(&I::from_i64(a * b, work)));
                        let v = prefactor
                            .mul(&numerator)
                            .div(&den[a.unsigned_abs() as usize])?
                            .div(&den[b.unsigned_abs() as usize])?;
                        rounded(&v, p)
                    })
                    .collect()
            })
            .collect::<Result<Option<Vec<_>>>>()?;
        if let Some(rows) = rows {
            return Ok((rows.into_iter().flatten().collect(), arch));
        }
    }
    bail!("matrix pole rounding unresolved within the guard budget")
}
pub(super) fn prime_matrix(n: usize, cutoff: u64, l: &Float, p: u32) -> Result<Vec<Float>> {
    let (d, base) = dimensions(n, l, p)?;
    if cutoff > 10_000_000 {
        bail!("prime matrix exceeds the supported sieve budget");
    }
    let events = super::super::try_prime_powers_up_to(cutoff)?;
    for guard in [64, 128, 256, 512, 1024, 2048, 4096] {
        let work = 2 * base + guard;
        let length = I::from_float(l, work)?;
        let pi = I::pi(work);
        let one = I::from_i64(1, work);
        let two = I::from_i64(2, work);
        let mut sines = vec![I::from_i64(0, work); n + 1];
        let mut diag = sines.clone();
        for &(power, prime, _) in &events {
            let r = one.sub(&I::from_u64(power, work).ln()?.div(&length)?);
            let weight = I::from_u64(prime, work)
                .ln()?
                .div(&I::from_u64(power, work).sqrt()?)?;
            for k in 0..=n {
                let phase = pi.mul(&I::from_u64((2 * k) as u64, work)).mul(&r);
                let sine = if k == 0 {
                    I::from_i64(0, work)
                } else {
                    phase.sin()
                };
                sines[k] = sines[k].add(&weight.mul(&sine));
                diag[k] = diag[k].add(&two.mul(&r).mul(&weight).mul(&phase.cos()));
            }
        }
        let signed = |m: i64| {
            let v = sines[m.unsigned_abs() as usize].clone();
            if m < 0 {
                v.neg()
            } else {
                v
            }
        };
        let mut out = vec![Float::with_val(p, 0); d * d];
        let mut resolved = true;
        'rows: for row in 0..d {
            let a = row as i64 - n as i64;
            for col in row..d {
                let b = col as i64 - n as i64;
                let v = if a == b {
                    diag[a.unsigned_abs() as usize].clone()
                } else {
                    signed(a)
                        .sub(&signed(b))
                        .div(&pi.mul(&I::from_i64(a - b, work)))?
                };
                let Some(v) = rounded(&v, p)? else {
                    resolved = false;
                    break 'rows;
                };
                out[row * d + col] = v.clone();
                out[col * d + row] = v;
            }
        }
        if resolved {
            return Ok(out);
        }
    }
    bail!("prime matrix rounding unresolved within the guard budget")
}
pub(super) fn total(c: &ComputedCcmMatrixComponents, p: u32) -> Result<Vec<Float>> {
    if !(64..=1_000_000).contains(&p)
        || c.pole.is_empty()
        || c.pole.len() != c.archimedean.len()
        || c.pole.len() != c.prime.len()
    {
        bail!("invalid component shape or precision");
    }
    let d = c.pole.len().isqrt();
    if d % 2 != 1 || d.checked_mul(d) != Some(c.pole.len()) || d > 8193 {
        bail!("components must have the same square odd shape");
    }
    c.pole
        .iter()
        .zip(&c.archimedean)
        .zip(&c.prime)
        .map(|((a, b), c)| {
            if [a, b, c]
                .iter()
                .any(|v| !v.is_finite() || v.prec() > 1_000_000)
            {
                bail!("invalid component point");
            }
            let terms = [a.clone(), -b.clone(), -c.clone()];
            let value = Float::with_val(p, Float::sum(terms.iter()));
            if !value.is_finite() {
                bail!("component sum overflow");
            }
            if value.is_zero() {
                let lo = Float::with_val_round(p, Float::sum(terms.iter()), Round::Down).0;
                let hi = Float::with_val_round(p, Float::sum(terms.iter()), Round::Up).0;
                if !lo.is_zero() || !hi.is_zero() {
                    bail!("nonzero component sum underflow");
                }
            }
            if value.get_exp() == Some(rug::float::exp_min())
                && Float::with_val_round(p, Float::sum(terms.iter()), Round::Zero)
                    .0
                    .is_zero()
            {
                bail!("component sum loses precision at exponent floor");
            }
            Ok(value)
        })
        .collect()
}

/// Exact actions of the stored component matrices; reconstructed prime includes
/// differences between the stored Tau and these independently staged components.
pub(super) fn component_actions(
    n: usize,
    l: &Float,
    p: u32,
    t: &ComputedArchimedeanIntegrals,
    tau: &[Float],
    v: &[Float],
) -> Result<Vec<(Float, Float, Float)>> {
    let (d, base) = dimensions(n, l, p)?;
    if tau.len() != d * d
        || v.len() != d
        || tau
            .iter()
            .chain(v)
            .any(|x| !x.is_finite() || x.prec() > 1_000_000)
    {
        bail!("invalid retained component-action source");
    }
    let base = base.max(tau.iter().chain(v).map(Float::prec).max().unwrap_or(base));
    rational_stage_budget(d, base, v.iter(), 1)?;
    let (pole, arch) = pole_arch(n, l, p, t)?;
    let active_rows = rayon::current_num_threads().min(d);
    for row in 0..d {
        let range = row * d..(row + 1) * d;
        rational_stage_budget(
            d,
            base,
            tau[range.clone()]
                .iter()
                .chain(&pole[range.clone()])
                .chain(&arch[range])
                .chain(v),
            active_rows,
        )?;
    }
    let v = v
        .iter()
        .map(|x| x.to_rational().unwrap())
        .collect::<Vec<_>>();
    (0..d)
        .into_par_iter()
        .map(|row| {
            let mut a = Rational::from(0);
            let mut b = Rational::from(0);
            let mut c = Rational::from(0);
            for (col, weight) in v.iter().enumerate() {
                let k = row * d + col;
                let po = pole[k].to_rational().unwrap();
                let ar = arch[k].to_rational().unwrap();
                a += po.clone() * weight;
                b -= ar.clone() * weight;
                c += (tau[k].to_rational().unwrap() - po + ar) * weight;
            }
            Ok((
                rational_round(a, p)?,
                rational_round(b, p)?,
                rational_round(c, p)?,
            ))
        })
        .collect()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn retained_component_reconstruction_preserves_exact_cancelled_total() {
        let p = 128;
        let huge = Float::with_val(p, 1) << 400u32;
        let tau = vec![
            huge.clone(),
            Float::with_val(p, 1),
            -huge.clone(),
            Float::with_val(p, 1),
            Float::with_val(p, 2),
            Float::with_val(p, 1),
            -huge.clone(),
            Float::with_val(p, 1),
            huge,
        ];
        let v = vec![Float::with_val(p, 1); 3];
        let l = Float::with_val(p, 3);
        let zero = vec![Float::with_val(p, 0); 2];
        let table = ComputedArchimedeanIntegrals {
            alpha: zero.clone(),
            beta: zero.clone(),
            gamma: zero,
        };
        let got = component_actions(1, &l, p, &table, &tau, &v).unwrap();
        let (pole, arch) = pole_arch(1, &l, p, &table).unwrap();
        for row in 0..3 {
            let exact = (0..3)
                .map(|col| {
                    tau[row * 3 + col].to_rational().unwrap()
                        - pole[row * 3 + col].to_rational().unwrap()
                        + arch[row * 3 + col].to_rational().unwrap()
                })
                .fold(Rational::from(0), |a, b| a + b);
            assert_eq!(got[row].2, Float::with_val(p, exact));
        }
        assert_eq!(Float::with_val(p, Float::sum(tau[..3].iter())), 1);
    }
    #[test]
    fn matrix_point_domains_reject_truncation_nonfinite_and_resource_overrides() {
        let p = 128;
        let l = Float::with_val(p, 3);
        let zero = vec![Float::with_val(p, 0); 1];
        let table = ComputedArchimedeanIntegrals {
            alpha: zero.clone(),
            beta: zero.clone(),
            gamma: zero.clone(),
        };
        assert!(pole_arch(1, &l, p, &table).is_err());
        assert!(prime_matrix(usize::MAX, 2, &l, p).is_err());
        assert!(prime_matrix(0, u64::MAX, &l, p).is_err());
        assert!(integrals(0, &l, p, &[Float::with_val(p, 0)], &[]).is_err());
        assert!(total(
            &ComputedCcmMatrixComponents {
                pole: zero.clone(),
                archimedean: vec![],
                prime: zero
            },
            p
        )
        .is_err());
        assert!(dimensions(4000, &l, 1_000_000).is_err());
    }
}

#[cfg(test)]
mod matrix_budget_tests {
    use super::*;
    #[test]
    fn dense_storage_is_charged_at_retained_precision() {
        let l = Float::with_val(64, 2);
        preflight(928, &l, 64).unwrap();
        assert!(matrix_workspace_bytes(1857, 64) < (2u128 << 30));
        assert!(preflight(4096, &Float::with_val(1_000_000, 2), 1_000_000).is_err());
    }
}
