//! Worker-local endpoint storage for the original directed integral terms.
//!
//! This does not change the products, their rounding, or the final MPFR sum.
//! In particular, zero signs and corner traversal match `MpfrInterval::mul`.
use super::*;
use rug::{ops::AssignRound, Assign};

type Endpoints<'a> = (&'a Float, &'a Float);

fn endpoints(value: &I) -> Endpoints<'_> {
    (value.lower(), value.upper())
}

fn valid(value: Endpoints<'_>, precision: u32) -> bool {
    value.0.prec() == precision
        && value.1.prec() == precision
        && value.0.is_finite()
        && value.1.is_finite()
        && value.0 <= value.1
}

struct Term {
    lower: Float,
    upper: Float,
}

impl Term {
    fn new(precision: u32) -> Self {
        Self {
            lower: Float::new(precision),
            upper: Float::new(precision),
        }
    }

    fn endpoints(&self) -> Endpoints<'_> {
        (&self.lower, &self.upper)
    }

    fn invalidate(&mut self) {
        self.lower.assign(rug::float::Special::Nan);
        self.upper.assign(rug::float::Special::Nan);
    }

    fn multiply(&mut self, a: Endpoints<'_>, b: Endpoints<'_>, work: &mut ProductScratch) {
        let p = self.lower.prec();
        if !valid(a, p) || !valid(b, p) {
            self.invalidate();
            return;
        }
        let left = [a.0, a.1];
        let right = [b.0, b.1];
        let left_count =
            usize::from(!(a.0 == a.1 && a.0.is_sign_negative() == a.1.is_sign_negative())) + 1;
        let right_count =
            usize::from(!(b.0 == b.1 && b.0.is_sign_negative() == b.1.is_sign_negative())) + 1;
        let mut first = true;
        for &x in &left[..left_count] {
            for &y in &right[..right_count] {
                work.down.assign_round(x * y, Round::Down);
                work.up.assign_round(x * y, Round::Up);
                if first || work.down.total_cmp(&self.lower).is_lt() {
                    std::mem::swap(&mut self.lower, &mut work.down);
                }
                if first || work.up.total_cmp(&self.upper).is_gt() {
                    std::mem::swap(&mut self.upper, &mut work.up);
                }
                first = false;
            }
        }
        if !valid(self.endpoints(), p) {
            self.invalidate();
        }
    }

    fn add(&mut self, a: Endpoints<'_>, b: Endpoints<'_>) {
        let p = self.lower.prec();
        if !valid(a, p) || !valid(b, p) {
            self.invalidate();
            return;
        }
        self.lower.assign_round(a.0 + b.0, Round::Down);
        self.upper.assign_round(a.1 + b.1, Round::Up);
        if !valid(self.endpoints(), p) {
            self.invalidate();
        }
    }
}

struct ProductScratch {
    down: Float,
    up: Float,
}

struct Buffers {
    precision: u32,
    alpha: Vec<Term>,
    beta: Vec<Term>,
    gamma: Vec<Term>,
    first: Term,
    second: Term,
    product: ProductScratch,
}

impl Buffers {
    fn new(precision: u32) -> Self {
        Self {
            precision,
            alpha: Vec::new(),
            beta: Vec::new(),
            gamma: Vec::new(),
            first: Term::new(precision),
            second: Term::new(precision),
            product: ProductScratch {
                down: Float::new(precision),
                up: Float::new(precision),
            },
        }
    }

    fn resize(&mut self, points: usize) {
        for terms in [&mut self.alpha, &mut self.beta, &mut self.gamma] {
            terms.resize_with(points, || Term::new(self.precision));
        }
    }

    fn alpha(&mut self, index: usize, weight: &I, sin: &I, combined: &I) {
        self.first
            .multiply(endpoints(weight), endpoints(sin), &mut self.product);
        self.alpha[index].multiply(
            self.first.endpoints(),
            endpoints(combined),
            &mut self.product,
        );
    }

    fn beta(&mut self, index: usize, weight: &I, h: &I, cos: &I, combined: &I) {
        self.first
            .multiply(endpoints(weight), endpoints(h), &mut self.product);
        self.second
            .multiply(self.first.endpoints(), endpoints(cos), &mut self.product);
        self.beta[index].multiply(
            self.second.endpoints(),
            endpoints(combined),
            &mut self.product,
        );
    }

    fn gamma_product(&mut self, difference: &I, combined: &I) {
        self.first.multiply(
            endpoints(difference),
            endpoints(combined),
            &mut self.product,
        );
    }

    fn gamma_finish(&mut self, index: usize, weight: &I, offset: &I) {
        self.second.add(self.first.endpoints(), endpoints(offset));
        self.gamma[index].multiply(
            endpoints(weight),
            self.second.endpoints(),
            &mut self.product,
        );
    }
}

/// A Rayon job owns this storage; it is never shared or retained in the cache.
/// A precision change discards endpoints rather than silently reusing old
/// destination precision. Every active slot is overwritten before summation.
#[derive(Default)]
pub(super) struct IntegralScratch {
    buffers: Option<Buffers>,
}

impl IntegralScratch {
    fn prepare(&mut self, points: usize, precision: u32) -> &mut Buffers {
        if self
            .buffers
            .as_ref()
            .is_none_or(|b| b.precision != precision)
        {
            self.buffers = Some(Buffers::new(precision));
        }
        let buffers = self.buffers.as_mut().expect("initialized integral scratch");
        buffers.resize(points);
        buffers
    }
}

fn sum_terms(values: &[Term], precision: u32) -> Result<I> {
    for value in values {
        if !valid(value.endpoints(), precision) {
            return Err(xc_numerics::interval::IntervalError::Invalid(
                "invalid or unrepresentable MPFR interval".into(),
            )
            .into());
        }
    }
    Ok(I::new(
        Float::with_val_round(
            precision,
            Float::sum(values.iter().map(|v| &v.lower)),
            Round::Down,
        )
        .0,
        Float::with_val_round(
            precision,
            Float::sum(values.iter().map(|v| &v.upper)),
            Round::Up,
        )
        .0,
    )?)
}

pub(super) fn integral_with_scratch(
    n: usize,
    l: &I,
    nodes: &[Float],
    weights: &[Float],
    p: u32,
    scratch: &mut IntegralScratch,
) -> Result<[I; 3]> {
    validate_table(nodes, weights)?;
    let buffers = scratch.prepare(nodes.len(), p);
    let one = I::from_i64(1, p);
    let two = I::from_i64(2, p);
    let half = I::point(Float::with_val(p, 0.5));
    let pi = I::pi(p);
    for (index, (node, weight)) in nodes.iter().zip(weights).enumerate() {
        let h = I::from_float(node, p)?.add(&one).mul(&half);
        let x = l.mul(&h);
        let e = x.mul(&half).neg().exp();
        let rho = e.div(&exp_m1(&x.mul(&two).neg())?.neg())?;
        let coth = one.add(&two.div(&exp_m1(&x.mul(&two))?)?);
        let combined = rho.mul(&one.add(&x.mul(&half.sub(&coth))));
        let phase = pi.mul(&I::from_u64((2 * n) as u64, p)).mul(&h);
        let quadrant = exact_quadrant(node, n)?;
        let (sin, cos, difference_cos) = if let Some(q) = quadrant {
            let cos = I::from_i64([1, 0, -1, 0][q], p);
            (I::from_i64([0, 1, 0, -1][q], p), cos.clone(), cos.sub(&one))
        } else {
            let (sin, cos) = phase.sin_cos();
            (sin, cos, phase.mul(&half).sin().square().mul(&two).neg())
        };
        let weight = I::from_float(weight, p)?;
        buffers.alpha(index, &weight, &sin, &combined);
        buffers.beta(index, &weight, &h, &cos, &combined);
        let difference = difference_cos.sub(&exp_m1(&x.mul(&half).neg())?);
        buffers.gamma_product(&difference, &combined);
        let offset = x.mul(&e).mul(&rho).mul(&half);
        buffers.gamma_finish(index, &weight, &offset);
    }
    Ok([
        sum_terms(&buffers.alpha, p)?.div(&two.mul(&pi))?,
        sum_terms(&buffers.beta, p)?.mul(&half),
        sum_terms(&buffers.gamma, p)?
            .mul(&half)
            .add(&one.div(&two.mul(&sinh(l)?))?),
    ])
}

pub(super) fn integral_shared_with_scratch(
    n: usize,
    nodes: &[Float],
    rule: &SharedRule,
    p: u32,
    scratch: &mut IntegralScratch,
) -> Result<[I; 3]> {
    let buffers = scratch.prepare(nodes.len().min(rule.factors.len()), p);
    let SharedRule {
        one, two, half, pi, ..
    } = rule;
    for (index, (node, factor)) in nodes.iter().zip(&rule.factors).enumerate() {
        let phase = pi.mul(&I::from_u64((2 * n) as u64, p)).mul(&factor.h);
        let quadrant = exact_quadrant(node, n)?;
        let (sin, cos, difference_cos) = if let Some(q) = quadrant {
            let cos = I::from_i64([1, 0, -1, 0][q], p);
            (I::from_i64([0, 1, 0, -1][q], p), cos.clone(), cos.sub(one))
        } else {
            let (sin, cos) = phase.sin_cos();
            (sin, cos, phase.mul(half).sin().square().mul(two).neg())
        };
        buffers.alpha(index, &factor.weight, &sin, &factor.combined);
        buffers.beta(index, &factor.weight, &factor.h, &cos, &factor.combined);
        let difference = difference_cos.sub(&factor.decay_difference);
        buffers.gamma_product(&difference, &factor.combined);
        buffers.gamma_finish(index, &factor.weight, &factor.gamma_offset);
    }
    Ok([
        sum_terms(&buffers.alpha, p)?.div(&two.mul(pi))?,
        sum_terms(&buffers.beta, p)?.mul(half),
        sum_terms(&buffers.gamma, p)?
            .mul(half)
            .add(&rule.gamma_tail),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_float(actual: &Float, expected: &Float) {
        assert_eq!(actual, expected);
        assert_eq!(actual.prec(), expected.prec());
        assert_eq!(actual.is_sign_negative(), expected.is_sign_negative());
    }

    fn assert_integrals(actual: &[I; 3], expected: &[I; 3]) {
        for (actual, expected) in actual.iter().zip(expected) {
            assert_float(actual.lower(), expected.lower());
            assert_float(actual.upper(), expected.upper());
        }
    }

    #[test]
    fn scratch_products_preserve_directed_endpoints_precision_and_zero_signs() {
        for p in [64, 128, 512, 4281, 6772] {
            let zero = Float::with_val(p, 0);
            let tiny = Float::with_val(p, 1) >> 2000u32;
            let cases = [
                I::point(zero.clone()),
                I::point(-zero.clone()),
                I::new(-zero.clone(), zero).unwrap(),
                I::point(tiny.clone()),
                I::new(-tiny.clone(), tiny).unwrap(),
                I::from_i64(-3, p),
                I::new(Float::with_val(p, -3), Float::with_val(p, 2)).unwrap(),
                I::new(Float::with_val(p, 2), Float::with_val(p, 3)).unwrap(),
                I::pi(p),
            ];
            let mut term = Term::new(p);
            let mut work = ProductScratch {
                down: Float::new(p),
                up: Float::new(p),
            };
            for a in &cases {
                for b in &cases {
                    term.multiply(endpoints(a), endpoints(b), &mut work);
                    let expected = a.mul(b);
                    assert_float(&term.lower, expected.lower());
                    assert_float(&term.upper, expected.upper());
                    term.add(endpoints(a), endpoints(b));
                    let expected = a.add(b);
                    assert_float(&term.lower, expected.lower());
                    assert_float(&term.upper, expected.upper());
                }
            }
            term.multiply(
                endpoints(&cases[0]),
                endpoints(&I::from_i64(1, p + 1)),
                &mut work,
            );
            assert!(term.lower.is_nan() && term.upper.is_nan());
            assert_eq!(term.lower.prec(), p);
        }
    }

    #[test]
    fn scratch_integrals_match_original_through_cancellation_and_repeated_rules() {
        let resources = xc_core::RunResources::new(1 << 28).unwrap();
        let _scope = resources.enter();
        let mut scratch = IntegralScratch::default();
        for (source_precision, p) in [(64, 128), (128, 512), (3418, 4281), (6772, 6772), (64, 128)]
        {
            let mut nodes = [-0.75, -0.5, -0.25, -0.0, 0.0, 0.25, 0.5, 0.75]
                .map(|x| Float::with_val(source_precision, x))
                .to_vec();
            nodes.push(Float::with_val(source_precision, 1) >> 400u32);
            nodes.push(-Float::with_val(source_precision, 1) >> 400u32);
            let weights = (0..nodes.len())
                .map(|i| Float::with_val(source_precision, i + 1) / 16u32)
                .collect::<Vec<_>>();
            for count in [nodes.len(), 3, nodes.len()] {
                let nodes = &nodes[..count];
                let weights = &weights[..count];
                for length in [1, 3] {
                    let l = I::from_i64(length, p);
                    let shared = prepare_shared_rule(&l, nodes, weights, p).unwrap();
                    for mode in [0, 1, 4, 17] {
                        let expected = super::super::integral(mode, &l, nodes, weights, p).unwrap();
                        let actual =
                            integral_with_scratch(mode, &l, nodes, weights, p, &mut scratch)
                                .unwrap();
                        assert_integrals(&actual, &expected);
                        let actual =
                            integral_shared_with_scratch(mode, nodes, &shared, p, &mut scratch)
                                .unwrap();
                        let original_shared =
                            super::super::integral_shared(mode, nodes, &shared, p).unwrap();
                        assert_integrals(&actual, &original_shared);
                        assert_integrals(&actual, &expected);
                    }
                }
            }
        }
    }

    #[test]
    fn scratch_sum_preserves_ordered_float_sum_and_cancellation() {
        for p in [64, 4281] {
            let mut terms = Vec::new();
            let mut original = Vec::new();
            for value in [1, -1, 0, -3, 3] {
                let v = I::from_i64(value, p);
                let mut term = Term::new(p);
                term.lower.assign(v.lower());
                term.upper.assign(v.upper());
                terms.push(term);
                original.push(v);
            }
            let actual = sum_terms(&terms, p).unwrap();
            let expected = sum(&original, p).unwrap();
            assert_float(actual.lower(), expected.lower());
            assert_float(actual.upper(), expected.upper());
        }
    }

    #[test]
    fn scratch_invalid_products_preserve_original_sum_failure() {
        let p = 128;
        let one = I::from_i64(1, p);
        let huge = I::point(Float::with_val(p, 1) << (rug::float::exp_max() - 1) as u32);
        for (a, b) in [(one, I::from_i64(1, p + 1)), (huge, I::from_i64(4, p))] {
            let expected = a.mul(&b);
            let mut term = Term::new(p);
            let mut work = ProductScratch {
                down: Float::new(p),
                up: Float::new(p),
            };
            term.multiply(endpoints(&a), endpoints(&b), &mut work);
            assert_eq!(
                sum_terms(&[term], p).unwrap_err().to_string(),
                sum(&[expected], p).unwrap_err().to_string(),
            );
        }
    }

    #[test]
    #[ignore = "supervised allocation-reuse measurement; not a claim-scale timing estimate"]
    fn measure_uflow_integral_scratch() {
        use std::{hint::black_box, time::Instant};
        let resources = xc_core::RunResources::new(1 << 28).unwrap();
        let _scope = resources.enter();
        for p in [512, 4281] {
            let points = 65;
            let modes = 16;
            let (nodes, weights) = xc_numerics::quadrature::try_gauss_legendre_nodes_scheduled(
                points,
                p,
                xc_numerics::quadrature::CacheMode::Off,
                xc_numerics::hp_runtime::plan_gl_precompute(&[points], p).root_schedule(points),
            )
            .unwrap();
            let l = I::from_i64(3, p);
            let shared = prepare_shared_rule(&l, &nodes, &weights, p).unwrap();
            let mut scratch = IntegralScratch::default();
            for mode in 0..=modes {
                assert_integrals(
                    &integral_shared_with_scratch(mode, &nodes, &shared, p, &mut scratch).unwrap(),
                    &super::super::integral_shared(mode, &nodes, &shared, p).unwrap(),
                );
            }
            for shared_route in [false, true] {
                let mut times = [Vec::new(), Vec::new()];
                for repetition in 0..7 {
                    for route in if repetition % 2 == 0 { [0, 1] } else { [1, 0] } {
                        let mut scratch = IntegralScratch::default();
                        let started = Instant::now();
                        for mode in 0..=modes {
                            black_box(
                                match (route, shared_route) {
                                    (0, false) => {
                                        super::super::integral(mode, &l, &nodes, &weights, p)
                                    }
                                    (0, true) => {
                                        super::super::integral_shared(mode, &nodes, &shared, p)
                                    }
                                    (_, false) => integral_with_scratch(
                                        mode,
                                        &l,
                                        &nodes,
                                        &weights,
                                        p,
                                        &mut scratch,
                                    ),
                                    (_, true) => integral_shared_with_scratch(
                                        mode,
                                        &nodes,
                                        &shared,
                                        p,
                                        &mut scratch,
                                    ),
                                }
                                .unwrap(),
                            );
                        }
                        times[route].push(started.elapsed().as_secs_f64());
                    }
                }
                println!("integral scratch raw: p={p}, shared={shared_route}, points={points}, modes={}, original={:?}, scratch={:?}", modes + 1, times[0], times[1]);
                for times in &mut times {
                    times.sort_by(f64::total_cmp);
                }
                println!("integral scratch median: p={p}, shared={shared_route}, original={:.6}s, scratch={:.6}s, speedup={:.4}x; unchanged endpoint arithmetic; table/rule preparation excluded, scratch initialization included; not a whole-claim estimate", times[0][3], times[1][3], times[0][3] / times[1][3]);
            }
        }
    }
}
