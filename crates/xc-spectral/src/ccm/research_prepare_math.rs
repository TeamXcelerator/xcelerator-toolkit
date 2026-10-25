//! Explicit finite point stages for automatic polynomial preparation.
//! Each transcendental output requires directed bounds to agree after rounding.
//! Exact dyadic form splitting is separate from continuum/source accuracy.
use anyhow::{bail, ensure, Result};
use num_bigint::{BigInt, Sign};
use rayon::prelude::*;
use rug::{float::Round, ops::Pow, Float, Integer, Rational};
use xc_numerics::mpfr_interval::MpfrInterval as I;
pub(super) const RATIONAL_BIT_BUDGET: u64 = 67_108_864;

fn preparation_worker_limit(pool: usize, run: usize, configured: Option<&str>) -> Result<usize> {
    let requested = match configured {
        Some(text) => {
            ensure!(
                !text.is_empty() && text.bytes().all(|b| b.is_ascii_digit()),
                "XC_RESEARCH_PREP_WORKERS must be a positive integer"
            );
            let value = text.parse::<usize>()?;
            ensure!(value > 0, "XC_RESEARCH_PREP_WORKERS must be positive");
            value
        }
        None => 16,
    };
    ensure!(pool > 0 && run > 0, "preparation worker limit is zero");
    Ok(requested.min(pool).min(run))
}

/// Bound active preparation tasks without constructing another thread pool.
/// Product engines continue to use the caller's independently bounded pool.
pub(super) fn preparation_workers() -> Result<usize> {
    let configured = match std::env::var("XC_RESEARCH_PREP_WORKERS") {
        Ok(text) => Some(text),
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => return Err(anyhow::anyhow!("invalid XC_RESEARCH_PREP_WORKERS: {error}")),
    };
    let run =
        xc_core::RunResources::current().map_or(usize::MAX, |resources| resources.worker_limit());
    preparation_worker_limit(rayon::current_num_threads(), run, configured.as_deref())
}

/// Ordered output, at most `workers` simultaneous tasks, no nested worker pool.
pub(super) fn preparation_map<T: Send>(
    count: usize,
    workers: usize,
    compute: impl Fn(usize) -> T + Sync,
) -> Vec<T> {
    if count == 0 {
        return Vec::new();
    }
    let chunk = count.div_ceil(workers.max(1));
    (0..count.div_ceil(chunk))
        .into_par_iter()
        .map(|block| {
            (block * chunk..count.min((block + 1) * chunk))
                .map(&compute)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
        .into_iter()
        .flatten()
        .collect()
}
fn check(x: &Rational) -> Result<()> {
    ensure!(
        u64::from(x.numer().significant_bits()) + u64::from(x.denom().significant_bits())
            <= RATIONAL_BIT_BUDGET,
        "preparation exact rational exceeds bit budget"
    );
    Ok(())
}
fn source(x: &Float) -> Result<Rational> {
    super::certified_roots::boundary::rational_budget(std::iter::once(x), 1)?;
    let q = x.to_rational().unwrap();
    check(&q)?;
    Ok(q)
}
pub(super) fn round(x: &Rational, p: u32) -> Result<Float> {
    check(x)?;
    let v = Float::with_val(p, x);
    ensure!(
        v.is_finite() && (!v.is_zero() || x == &0),
        "preparation point exceeds exponent range"
    );
    ensure!(
        v.get_exp() != Some(rug::float::exp_min())
            || !Float::with_val_round(p, x, Round::Zero).0.is_zero(),
        "preparation point partially underflows"
    );
    Ok(v)
}
fn rounded(x: &I, p: u32) -> Result<Option<Float>> {
    x.validate()?;
    let lo = Float::with_val(p, x.lower());
    let hi = Float::with_val(p, x.upper());
    Ok((lo == hi
        && lo.is_finite()
        && (!lo.is_zero() || (x.lower().is_zero() && x.upper().is_zero())))
    .then_some(lo))
}
fn shape(n: usize, p: u32) -> Result<()> {
    ensure!(
        n > 0
            && n.checked_mul(n).is_some()
            && n.checked_mul(2)
                .and_then(|value| value.checked_add(1))
                .is_some_and(|value| u32::try_from(value).is_ok())
            && (64..=995_904).contains(&p),
        "preparation dimension/precision exceeds supported domain"
    );
    Ok(())
}
pub(super) fn lattice(k: usize, n: usize) -> Rational {
    Rational::from((k * k, n * n))
}
pub(super) fn square_sum(x: &[Float], p: u32) -> Result<Float> {
    let bits = x.iter().try_fold(0u128, |sum, value| -> Result<_> {
        // Inspect exponent span before allocating an exact numerator or
        // denominator. A tiny transform value need not share the ordinate
        // coordinates' exponent range.
        super::certified_roots::boundary::rational_budget(std::iter::once(value), 1)?;
        let source_bits = 2 * u128::from(value.prec())
            + u128::from(i64::from(value.get_exp().unwrap_or(0)).unsigned_abs())
            + 8;
        sum.checked_add(source_bits)
            .ok_or_else(|| anyhow::anyhow!("preparation exact square size overflow"))
    })?;
    let bytes = u64::try_from(
        bits.checked_mul(4)
            .and_then(|value| value.checked_add(x.len() as u128 * 256 + 4096))
            .ok_or_else(|| anyhow::anyhow!("preparation exact square size overflow"))?,
    )?;
    let resources = match xc_core::RunResources::current() {
        Some(resources) => resources,
        None => xc_core::RunResources::new(
            super::capture_runtime::CaptureResourcePolicy::from_environment()?
                .maximum_working_bytes,
        )?,
    };
    let _workspace = resources.try_reserve_working(bytes).map_err(|error| {
        anyhow::anyhow!("preparation exact square exceeds remaining workspace: {error}")
    })?;
    let mut q = Rational::new();
    for x in x {
        let v = source(x)?;
        q += v.square();
        check(&q)?;
    }
    round(&q, p)
}
pub(super) fn coordinates(t: &[Float], l: &Float, n: usize, p: u32) -> Result<Vec<Float>> {
    shape(n, p)?;
    ensure!(
        l > &0 && !t.is_empty() && t.len() <= 1_000_000,
        "invalid preparation coordinates"
    );
    let l = source(l)?;
    t.iter()
        .map(|t| {
            let q = source(t)? * &l / (2 * n);
            for guard in [64, 128, 256, 512, 1024, 2048, 4096] {
                let w = p + guard;
                let value = I::from_rational(&q, w).div(&I::pi(w))?.square();
                if let Some(v) = rounded(&value, p)? {
                    return Ok(v);
                }
            }
            bail!("preparation coordinate rounding unresolved")
        })
        .collect()
}
pub(super) fn basis(z: &[Float], l: &Float, n: usize, d: usize, p: u32) -> Result<Vec<Float>> {
    shape(n, p)?;
    let workers = preparation_workers()?;
    ensure!(
        (1..=65).contains(&d) && d <= n + 1 && z.len() >= n + 1 - d && l > &0,
        "invalid preparation basis shape"
    );
    let prefix = n + 1 - d;
    let roots = z[..prefix].iter().map(source).collect::<Result<Vec<_>>>()?;
    let length = source(l)?;
    let mut facts = vec![Integer::from(1); 2 * n + 1];
    for k in 1..=2 * n {
        facts[k] = facts[k - 1].clone() * k;
    }
    let power = Integer::from(n).pow(2 * n as u32);
    // Each lattice mode k owns rows n-k and n+k and depends on nothing else,
    // so modes run on workers; the rows are assembled, and the first error
    // reported, in k order, exactly as the serial loop did.
    xc_numerics::mpfr_interval::ensure_uniform_exponent_range()?;
    let mut v = vec![Float::with_val(p, 0); (2 * n + 1) * d];
    for first in (0..=n).step_by(workers) {
        let rows = (first..(n + 1).min(first + workers))
            .into_par_iter()
            .map(|k| -> Result<Vec<Float>> {
                let a = lattice(k, n);
                let mut q = Rational::from((power.clone(), facts[n - k].clone() * &facts[n + k]));
                if k % 2 == 1 {
                    q = -q;
                }
                for r in &roots {
                    q *= a.clone() - r;
                    check(&q)?;
                }
                let mut row = Vec::with_capacity(d);
                for _ in 0..d {
                    let mut value = None;
                    for guard in [64, 128, 256, 512, 1024, 2048, 4096] {
                        let w = p + guard;
                        let r =
                            I::from_rational(&q, w).div(&I::from_rational(&length, w).sqrt()?)?;
                        if let Some(r) = rounded(&r, p)? {
                            value = Some(r);
                            break;
                        }
                    }
                    row.push(
                        value.ok_or_else(|| {
                            anyhow::anyhow!("preparation basis rounding unresolved")
                        })?,
                    );
                    q *= &a;
                    check(&q)?;
                }
                Ok(row)
            })
            .collect::<Vec<_>>();
        for (offset, row) in rows.into_iter().enumerate() {
            let k = first + offset;
            for (j, value) in row?.into_iter().enumerate() {
                v[(n + k) * d + j] = value.clone();
                v[(n - k) * d + j] = value;
            }
        }
    }
    Ok(v)
}
fn form_shape(dim: usize, d: usize, matrix_entries: usize, basis_entries: usize) -> Result<()> {
    ensure!(
        dim > 0
            && d > 0
            && d <= 65
            && dim.checked_mul(dim) == Some(matrix_entries)
            && dim.checked_mul(d) == Some(basis_entries),
        "preparation form shape mismatch"
    );
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct DyadicSpan {
    upper: u64,
    denominator: u64,
}

impl DyadicSpan {
    fn of(values: &[Float]) -> Result<Self> {
        let mut result = Self {
            upper: 0,
            denominator: 0,
        };
        for value in values {
            ensure!(value.is_finite(), "nonfinite preparation form source");
            if let Some(exponent) = value.get_exp() {
                result.upper = result.upper.max(i64::from(exponent).max(0) as u64);
                result.denominator = result
                    .denominator
                    .max((i64::from(value.prec()) - i64::from(exponent)).max(0) as u64);
            }
        }
        Ok(result)
    }

    fn bits(self) -> u64 {
        self.upper + self.denominator
    }
}

#[derive(Clone, Copy, Debug)]
struct DyadicFormPlan {
    a: DyadicSpan,
    v: DyadicSpan,
    av_bits: u64,
    av_denominator: u64,
    gram_bits: u64,
    gram_denominator: u64,
    actual_bits: u64,
    actual_denominator: u64,
}

impl DyadicFormPlan {
    fn new(matrix: &[Float], v: &[Float], dim: usize, p: u32) -> Result<Option<Self>> {
        let a = DyadicSpan::of(matrix)?;
        let v_span = DyadicSpan::of(v)?;
        let log = u64::from(usize::BITS - (dim - 1).leading_zeros());
        let av_bits = a.bits() + v_span.bits() + log + 1;
        let av_denominator = a.denominator + v_span.denominator + 1;
        let gram_bits = 2 * v_span.bits() + log;
        let gram_denominator = 2 * v_span.denominator;
        let actual_bits = a.bits() + 2 * v_span.bits() + 2 * log + 1;
        let actual_denominator = a.denominator + 2 * v_span.denominator + 1;
        let plan = Self {
            a,
            v: v_span,
            av_bits,
            av_denominator,
            gram_bits,
            gram_denominator,
            actual_bits,
            actual_denominator,
        };
        // These absolute envelopes cover every original ordered prefix, not
        // only the final value. Unsupported ranges retain the ordered GMP path.
        let pairs = [
            (a.bits(), a.denominator),
            (v_span.bits(), v_span.denominator),
            (av_bits, av_denominator),
            (gram_bits, gram_denominator),
            (actual_bits, actual_denominator + 1),
        ];
        if pairs
            .iter()
            .any(|&(bits, denominator)| bits + denominator + 1 > RATIONAL_BIT_BUDGET)
            || !(64..=1_000_000).contains(&p)
        {
            return Ok(None);
        }
        if !plan.normal_range(p, rug::float::exp_min(), rug::float::exp_max()) {
            return Ok(None);
        }
        // Preserve source conversion preconditions before selecting the new
        // branch. A rejection falls back so the original error order remains.
        for value in v.iter().chain(matrix) {
            if super::certified_roots::boundary::rational_budget(std::iter::once(value), 1).is_err()
            {
                return Ok(None);
            }
        }
        Ok(Some(plan))
    }

    fn normal_range(self, p: u32, minimum_exponent: i32, maximum_exponent: i32) -> bool {
        let minimum = 1i64 - self.gram_denominator.max(self.actual_denominator + 1) as i64;
        let maximum = (self.gram_bits as i64 + 1 - self.gram_denominator as i64)
            .max(self.actual_bits as i64 - self.actual_denominator as i64);
        minimum_exponent <= 1
            && minimum >= i64::from(minimum_exponent)
            && maximum <= i64::from(maximum_exponent)
            && u64::from(p).min(self.gram_bits.max(self.actual_bits))
                <= maximum_exponent.max(0) as u64
    }

    fn workspace(self, dim: usize, d: usize, p: u32, workers: usize) -> Result<u64> {
        // Match the public shape domain before multiplying allocation counts.
        // Together with the admitted bit envelopes this bounds every u128
        // expression below, including on a 64-bit target.
        ensure!(
            dim > 0
                && dim.checked_mul(dim).is_some()
                && (1..=65).contains(&d)
                && (64..=1_000_000).contains(&p),
            "preparation exact form size overflow"
        );
        let n = dim as u128;
        let d = d as u128;
        let w = workers.min(dim).max(1) as u128;
        let pairs = d * (d + 1) / 2;
        // num-bigint 0.4.8 digit vectors grow geometrically. Using 32-bit
        // digit counts with eight bytes per slot covers doubled capacity for
        // either supported native digit width, plus carry/inline transitions.
        let integer = |bits: u64| -> u128 {
            u128::from(bits).div_ceil(32) * 8 + 16 + std::mem::size_of::<BigInt>() as u128
        };
        let product_bits = (self.a.bits() + 1 + self.v.bits())
            .max(2 * self.v.bits())
            .max(self.v.bits() + self.av_bits);
        let operand_words = u128::from(self.v.bits().max(self.av_bits)).div_ceil(32)
            + u128::from(self.a.bits().max(self.v.bits())).div_ceil(32)
            + 4;
        let depth = u128::from(128 - operand_words.max(1).leading_zeros());
        // Multiplication below caps the shorter operand at 256 u32 digits.
        // mac3's long/half-Karatsuba/Karatsuba branches then have <8*s live
        // recursive native limbs, with three headers per level. Toom3 is never
        // reached. Separate integer slots cover chunk, shifted partial, result,
        // and old/new reallocation exposure outside that recursive workspace.
        let multiplication = 8 * operand_words * 8
            + 12 * depth * std::mem::size_of::<BigInt>() as u128
            + 4 * integer(product_bits)
            + 2 * 8192 / 8;
        let persistent = n * d * integer(self.v.bits())
            + w * d * integer(self.av_bits)
            + pairs * (integer(self.gram_bits) + integer(self.actual_bits));
        let worker = 3 * integer(self.a.bits() + 1) + multiplication + (2 << 20);
        let float = u128::from(p).div_ceil(64) * 8 + 8 + std::mem::size_of::<Float>() as u128;
        let outputs = (2 * d * d + 4) * float
            + u128::from(p).div_ceil(32) * 4
            + u128::from(p).div_ceil(64) * 8
            + 32;
        // Vec/Result/pair/row headers and conversion's temporary u32 digits.
        let containers = pairs * std::mem::size_of::<(usize, usize)>() as u128
            + w * (2 * std::mem::size_of::<Vec<BigInt>>()
                + std::mem::size_of::<Result<Vec<BigInt>>>()
                + std::mem::size_of::<Result<()>>()) as u128;
        let source_bits = self.a.bits().max(self.v.bits());
        // Safe Float export owns one p-bit GMP Integer and a u32 digit Vec.
        // The shifted native BigInt may coexist with its prior allocation.
        let conversion = u128::from(source_bits).div_ceil(64) * 8
            + 16
            + u128::from(source_bits).div_ceil(32) * 4
            + 3 * integer(source_bits);
        u64::try_from(persistent + w * worker + outputs + containers + w * conversion)
            .map_err(|_| anyhow::anyhow!("preparation exact form size overflow"))
    }
}

/// Export a finite MPFR value onto its preadmitted common binary lattice.
/// The temporary GMP Integer holds only the source precision, not its full
/// denominator/exponent span. No raw MPFR representation is accessed.
fn dyadic_integer(value: &Float, span: DyadicSpan) -> Result<BigInt> {
    if value.is_zero() {
        return Ok(BigInt::ZERO);
    }
    let (integer, exponent) = value.to_integer_exp().unwrap();
    let digits = integer.to_digits::<u32>(rug::integer::Order::Lsf);
    let sign = if value.is_sign_negative() {
        Sign::Minus
    } else {
        Sign::Plus
    };
    let shift = usize::try_from(i64::from(exponent) + i64::try_from(span.denominator)?)?;
    Ok(BigInt::new(sign, digits) << shift)
}

fn dyadic_check(value: &BigInt, denominator: u64) -> Result<()> {
    let bits = value.bits();
    let count = if bits == 0 {
        1
    } else {
        let cancel = value.trailing_zeros().unwrap().min(denominator);
        bits - cancel + denominator - cancel + 1
    };
    ensure!(
        count <= RATIONAL_BIT_BUDGET,
        "preparation exact rational exceeds bit budget"
    );
    Ok(())
}

fn dyadic_multiply(left: &BigInt, right: &BigInt) -> BigInt {
    let (short, long) = if left.bits() <= right.bits() {
        (left, right)
    } else {
        (right, left)
    };
    if short.bits() <= 8192 {
        return short * long;
    }
    let mut result = BigInt::ZERO;
    let mut digits = short.iter_u32_digits();
    let mut shift = 0usize;
    loop {
        let chunk = digits.by_ref().take(256).collect::<Vec<_>>();
        if chunk.is_empty() {
            break;
        }
        let chunk = BigInt::new(short.sign(), chunk);
        result += (&chunk * long) << shift;
        shift += 8192;
    }
    result
}

fn dyadic_round(value: &BigInt, denominator: u64, p: u32) -> Result<Float> {
    dyadic_check(value, denominator)?;
    if value.bits() == 0 {
        return Ok(Float::with_val(p, 0));
    }
    let shift = value.bits().saturating_sub(u64::from(p));
    let mut magnitude = value.magnitude() >> usize::try_from(shift)?;
    if shift > 0
        && value.magnitude().bit(shift - 1)
        && (value.trailing_zeros().unwrap() < shift - 1 || magnitude.bit(0))
    {
        magnitude += 1u32;
    }
    let mut shift = shift;
    if magnitude.bits() > u64::from(p) {
        magnitude >>= 1usize;
        shift += 1;
    }
    let mut integer = Integer::from_digits(&magnitude.to_u32_digits(), rug::integer::Order::Lsf);
    if value.sign() == Sign::Minus {
        integer = -integer;
    }
    let mut result = Float::with_val(p, &integer);
    let exponent = i64::try_from(shift)? - i64::try_from(denominator)?;
    if exponent >= 0 {
        result <<= u32::try_from(exponent)?;
    } else {
        result >>= u32::try_from(-exponent)?;
    }
    ensure!(
        result.is_finite() && !result.is_zero(),
        "preparation point exceeds exponent range"
    );
    ensure!(
        result.get_exp() != Some(rug::float::exp_min())
            || i64::try_from(value.bits())? - i64::try_from(denominator)?
                >= i64::from(rug::float::exp_min()),
        "preparation point partially underflows"
    );
    Ok(result)
}

fn dyadic_forms(
    matrix: &[Float],
    basis: &[Float],
    dim: usize,
    d: usize,
    p: u32,
    resources: &xc_core::RunResources,
    requested_workers: usize,
) -> Result<Option<(Vec<Float>, Vec<Float>)>> {
    let Some(plan) = DyadicFormPlan::new(matrix, basis, dim, p)? else {
        return Ok(None);
    };
    let mut workers = requested_workers.min(dim).max(1);
    let workspace = loop {
        let bytes = plan.workspace(dim, d, p, workers)?;
        match resources.try_reserve_working(bytes) {
            Ok(reservation) => break reservation,
            Err(_) if workers > 1 => workers = workers.div_ceil(2),
            Err(_) => return Ok(None),
        }
    };
    // Source conversion is cheap beside the products. A single preallocated
    // vector avoids overlapping nested Result buffers and duplicate headers.
    let mut v = Vec::with_capacity(basis.len());
    for value in basis {
        v.push(dyadic_integer(value, plan.v)?);
    }
    xc_numerics::mpfr_interval::ensure_uniform_exponent_range()?;
    let mut pairs = Vec::with_capacity(d * (d + 1) / 2);
    for i in 0..d {
        for j in 0..=i {
            pairs.push((i, j));
        }
    }
    let mut sums = vec![(BigInt::ZERO, BigInt::ZERO); pairs.len()];
    let chunk = pairs.len().div_ceil(workers);
    for first in (0..dim).step_by(workers) {
        let rows = (first..dim.min(first + workers))
            .into_par_iter()
            .map(|r| -> Result<_> {
                let mut row = vec![BigInt::ZERO; d];
                for k in 0..dim {
                    let a = dyadic_integer(&matrix[r * dim + k], plan.a)?
                        + dyadic_integer(&matrix[k * dim + r], plan.a)?;
                    for j in 0..d {
                        row[j] += dyadic_multiply(&a, &v[k * d + j]);
                        dyadic_check(&row[j], plan.av_denominator)?;
                    }
                }
                Ok(row)
            })
            .collect::<Vec<_>>();
        for (offset, row) in rows.into_iter().enumerate() {
            let row = row?;
            let k = first + offset;
            let results = sums
                .par_chunks_mut(chunk)
                .enumerate()
                .map(|(block, sums)| -> Result<()> {
                    for (offset, (g, a)) in sums.iter_mut().enumerate() {
                        let (i, j) = pairs[block * chunk + offset];
                        *g += dyadic_multiply(&v[k * d + i], &v[k * d + j]);
                        *a += dyadic_multiply(&v[k * d + i], &row[j]);
                        dyadic_check(g, plan.gram_denominator)?;
                        dyadic_check(a, plan.actual_denominator)?;
                    }
                    Ok(())
                })
                .collect::<Vec<_>>();
            for result in results {
                result?;
            }
        }
    }
    drop(v);
    let mut gram = vec![Float::with_val(p, 0); d * d];
    let mut actual = gram.clone();
    for ((i, j), (g, a)) in pairs.into_iter().zip(sums) {
        let g = dyadic_round(&g, plan.gram_denominator, p)?;
        let a = dyadic_round(&a, plan.actual_denominator + 1, p)?;
        gram[i * d + j] = g.clone();
        gram[j * d + i] = g;
        actual[i * d + j] = a.clone();
        actual[j * d + i] = a;
    }
    drop(workspace);
    Ok(Some((gram, actual)))
}

pub(super) fn forms(
    matrix: &[Float],
    v: &[Float],
    dim: usize,
    d: usize,
    p: u32,
) -> Result<(Vec<Float>, Vec<Float>)> {
    form_shape(dim, d, matrix.len(), v.len())?;
    let workers = preparation_workers()?;
    let resources = match xc_core::RunResources::current() {
        Some(resources) => resources,
        None => {
            // The legacy path rejects a nonfinite source before reading the
            // standalone resource policy. Preserve that error precedence.
            ensure!(
                matrix.iter().chain(v).all(Float::is_finite),
                "nonfinite preparation form source"
            );
            xc_core::RunResources::new(
                super::capture_runtime::CaptureResourcePolicy::from_environment()?
                    .maximum_working_bytes,
            )?
        }
    };
    if let Some(result) = dyadic_forms(matrix, v, dim, d, p, &resources, workers)? {
        return Ok(result);
    }
    forms_gmp(matrix, v, dim, d, p, &resources, workers)
}

fn forms_gmp(
    matrix: &[Float],
    v: &[Float],
    dim: usize,
    d: usize,
    p: u32,
    resources: &xc_core::RunResources,
    workers: usize,
) -> Result<(Vec<Float>, Vec<Float>)> {
    let estimate = matrix
        .iter()
        .chain(v)
        .try_fold(0u128, |sum, x| -> Result<_> {
            ensure!(x.is_finite(), "nonfinite preparation form source");
            Ok(sum
                + u128::from(x.prec())
                + u128::from(i64::from(x.get_exp().unwrap_or(0)).unsigned_abs())
                + 2)
        })?;
    // Reserve the exact-form envelope against remaining shared capacity.
    // Merely comparing with the run's total cap would ignore live parents and
    // the preparation workspace already held by the caller.
    let bytes = u64::try_from(
        estimate
            .checked_mul(d as u128 + 4)
            .ok_or_else(|| anyhow::anyhow!("preparation exact form size overflow"))?
            .div_ceil(8),
    )?;
    let _workspace = resources.try_reserve_working(bytes).map_err(|error| {
        anyhow::anyhow!("preparation exact form exceeds remaining workspace: {error}")
    })?;
    let v = v.iter().map(source).collect::<Result<Vec<_>>>()?;
    // Rows of A*V and the (i, j) pairs are independent exact sums. Each keeps
    // the serial summation order and budget checks; the first failure in the
    // serial loop order is reported.
    xc_numerics::mpfr_interval::ensure_uniform_exponent_range()?;
    let pairs = (0..d)
        .flat_map(|i| (0..=i).map(move |j| (i, j)))
        .collect::<Vec<_>>();
    struct PairSum {
        gram: Rational,
        actual: Rational,
        error: Option<anyhow::Error>,
    }
    let mut sums = (0..pairs.len())
        .map(|_| PairSum {
            gram: Rational::new(),
            actual: Rational::new(),
            error: None,
        })
        .collect::<Vec<_>>();
    // Only one bounded row batch of AV is retained. Each pair still receives
    // its terms in increasing row order. Defer pair failures until all AV rows
    // have passed, preserving the original phase and pair error precedence.
    let pair_chunk = pairs.len().div_ceil(workers);
    for first in (0..dim).step_by(workers) {
        let rows = (first..dim.min(first + workers))
            .into_par_iter()
            .map(|r| -> Result<Vec<Rational>> {
                let mut row = vec![Rational::new(); d];
                for k in 0..dim {
                    let a: Rational =
                        (source(&matrix[r * dim + k])? + source(&matrix[k * dim + r])?) / 2;
                    for j in 0..d {
                        row[j] += a.clone() * &v[k * d + j];
                        check(&row[j])?;
                    }
                }
                Ok(row)
            })
            .collect::<Vec<_>>();
        for (offset, row) in rows.into_iter().enumerate() {
            let av = row?;
            let k = first + offset;
            sums.par_chunks_mut(pair_chunk)
                .enumerate()
                .for_each(|(block, sums)| {
                    for (offset, sum) in sums.iter_mut().enumerate() {
                        if sum.error.is_some() {
                            continue;
                        }
                        let (i, j) = pairs[block * pair_chunk + offset];
                        sum.gram += v[k * d + i].clone() * &v[k * d + j];
                        sum.actual += v[k * d + i].clone() * &av[j];
                        sum.error = check(&sum.gram).and_then(|()| check(&sum.actual)).err();
                    }
                });
        }
    }
    // Drop the exact basis before output expansion. The original rounded
    // values and symmetric Float arrays remain unchanged.
    drop(v);
    let values = sums
        .into_iter()
        .map(|sum| -> Result<(Float, Float)> {
            if let Some(error) = sum.error {
                return Err(error);
            }
            Ok((round(&sum.gram, p)?, round(&(sum.actual / 2), p)?))
        })
        .collect::<Vec<_>>();
    let mut gram = vec![Float::with_val(p, 0); d * d];
    let mut actual = gram.clone();
    for (&(i, j), value) in pairs.iter().zip(values) {
        let (g, a) = value?;
        gram[i * d + j] = g.clone();
        gram[j * d + i] = g;
        actual[i * d + j] = a.clone();
        actual[j * d + i] = a;
    }
    Ok((gram, actual))
}
pub(super) fn head(
    t: &[Float],
    z: &[Float],
    l: &Float,
    n: usize,
    d: usize,
    p: u32,
) -> Result<Vec<Float>> {
    shape(n, p)?;
    let workers = preparation_workers()?;
    ensure!(
        d > 0
            && d <= 65
            && d <= n + 1
            && t.len() == z.len()
            && t.len() >= n + 1 - d
            && t.len() <= 1_000_000,
        "invalid preparation head shape"
    );
    let prefix = n + 1 - d;
    let l = source(l)?;
    super::certified_roots::boundary::rational_budget(z.iter(), 1)?;
    let z = z.iter().map(source).collect::<Result<Vec<_>>>()?;
    // Stream exact coefficients: their aggregate size can exceed the budget
    // even though a single coefficient and the moment accumulators fit. Keep
    // the same exact factors and directed summation order at each guard tier.
    // Each ordinate's weight is formed on a worker, in bounded blocks; the
    // moments are accumulated serially in ordinate order, and the first
    // failure in that order is reported.
    xc_numerics::mpfr_interval::ensure_uniform_exponent_range()?;
    for guard in [64, 128, 256, 512, 1024, 2048, 4096] {
        let w = p + guard;
        let mut moments = vec![I::from_i64(0, w); 2 * d - 1];
        for first in (prefix..t.len()).step_by(workers) {
            let weights = (first..t.len().min(first + workers))
                .into_par_iter()
                .map(|j| -> Result<(I, I)> {
                    let theta = source(&t[j])? * &l / 2;
                    let mut factor = Rational::from(1);
                    for r in &z[..prefix] {
                        factor *= z[j].clone() - r;
                        check(&factor)?;
                    }
                    for k in 1..=n {
                        let denominator = z[j].clone() - lattice(k, n);
                        ensure!(
                            denominator != 0,
                            "stored preparation coordinate is a rational pole"
                        );
                        factor /= denominator;
                        check(&factor)?;
                    }
                    let factor = factor.square();
                    check(&factor)?;
                    let theta_i = I::from_rational(&theta, w);
                    let sinc = if theta == 0 {
                        I::from_i64(1, w)
                    } else {
                        theta_i.sin().div(&theta_i)?
                    };
                    Ok((
                        sinc.square().mul(&I::from_rational(&factor, w)),
                        I::from_rational(&z[j], w),
                    ))
                })
                .collect::<Vec<_>>();
            for value in weights {
                let (mut weight, z) = value?;
                for m in &mut moments {
                    *m = m.add(&weight);
                    weight = weight.mul(&z);
                }
            }
        }
        let values = moments
            .iter()
            .map(|x| rounded(x, p))
            .collect::<Result<Option<Vec<_>>>>()?;
        if let Some(values) = values {
            return Ok((0..d * d).map(|k| values[k / d + k % d].clone()).collect());
        }
    }
    bail!("preparation head rounding unresolved within guard budget")
}
/// Preserve the exact difference of the stored actual/head points by promoting
/// representation precision. The precision increase does not add source accuracy.
fn split_scalar_bytes(actual: &[Float], head: &[Float]) -> Result<u64> {
    let scratch = actual
        .iter()
        .zip(head)
        .map(|(a, h)| {
            let span = |x: &Float| {
                2 * u128::from(x.prec())
                    + u128::from(i64::from(x.get_exp().unwrap_or(0)).unsigned_abs())
                    + 2
            };
            let (a, h) = (span(a), span(h));
            (a + h).div_ceil(8) + 64 * a.max(h).div_ceil(8) + 2 * 96
        })
        .max()
        .unwrap_or(0);
    u64::try_from(scratch).map_err(|_| anyhow::anyhow!("preparation exact split size overflow"))
}

pub(super) fn split(
    actual: &[Float],
    head: &[Float],
    p: u32,
    resources: &xc_core::RunResources,
    reserve_output: impl FnOnce(u32) -> Result<xc_core::MemoryReservation>,
) -> Result<(u32, Vec<Float>, xc_core::MemoryReservation)> {
    ensure!(
        actual.len() == head.len(),
        "preparation tail shape mismatch"
    );
    // Adapt the established rational_point_vector_workspace scalar envelope:
    // two source owners plus 64 maximum-span scratch copies and their headers.
    // Only one difference is retained at a time. Metadata sizing deliberately
    // adds no source-domain rejection; original source/check errors occur in
    // their original order below, including after an earlier large target.
    let scratch = split_scalar_bytes(actual, head)?;
    let _scratch = resources.try_reserve_working(scratch).map_err(|error| {
        anyhow::anyhow!("preparation exact split exceeds remaining workspace: {error}")
    })?;
    let mut target = p;
    for (a, h) in actual.iter().zip(head) {
        let difference = source(a)? - source(h)?;
        target = target.max(difference.numer().significant_bits());
    }
    // The old vector conversion completed before this global check. Do not
    // return it early and mask a later original source-conversion error.
    ensure!(
        target <= 1_000_000,
        "exact preparation tail exceeds representation precision budget"
    );
    // The caller owns the complete promoted output envelope. Admit it before
    // allocating the first promoted Float; return its guard with the output.
    let output = reserve_output(target)?;
    let mut values = Vec::with_capacity(actual.len());
    for (a, h) in actual.iter().zip(head) {
        let difference = source(a)? - source(h)?;
        let value = round(&difference, target)?;
        ensure!(
            value.to_rational().as_ref() == Some(&difference),
            "tail split lost exact dyadic difference"
        );
        values.push(value);
    }
    Ok((target, values, output))
}
#[cfg(test)]
mod tests {
    use super::*;
    fn rational_from_dyadic(value: &BigInt, denominator: u64) -> Rational {
        let mut numerator =
            Integer::from_digits(&value.magnitude().to_u32_digits(), rug::integer::Order::Lsf);
        if value.sign() == Sign::Minus {
            numerator = -numerator;
        }
        Rational::from((
            numerator,
            Integer::from(1) << u32::try_from(denominator).unwrap(),
        ))
    }

    #[test]
    fn dyadic_export_keeps_exact_values_without_denominator_allocation() {
        for p in [64, 65, 127, 128, 3386, 6708] {
            let values = [-7, -1, 0, 1, 13]
                .into_iter()
                .enumerate()
                .map(|(index, value)| {
                    let mut x = Float::with_val(p, value);
                    if index.is_multiple_of(2) {
                        x <<= 701u32;
                    } else {
                        x >>= 1069u32;
                    }
                    x
                })
                .collect::<Vec<_>>();
            let span = DyadicSpan::of(&values).unwrap();
            for value in values {
                let integer = dyadic_integer(&value, span).unwrap();
                assert_eq!(
                    rational_from_dyadic(&integer, span.denominator),
                    value.to_rational().unwrap()
                );
            }
        }
    }

    #[test]
    fn dyadic_round_preserves_ties_carries_signs_zero_and_precision() {
        for p in [64u32, 65, 128, 3386, 6708] {
            for difference in [1u32, 2, 3, 4] {
                let base = (BigInt::from(1u32) << p as usize) - difference;
                for remainder in [0u32, 7, 8, 9, 15] {
                    for sign in [-1i32, 1] {
                        let value: BigInt = ((&base << 4usize) + remainder) * sign;
                        for denominator in [0u64, 17, u64::from(p) + 30] {
                            let expected =
                                round(&rational_from_dyadic(&value, denominator), p).unwrap();
                            let actual = dyadic_round(&value, denominator, p).unwrap();
                            assert_eq!(actual.prec(), expected.prec());
                            assert_eq!(actual.as_ord(), expected.as_ord());
                            assert_eq!(actual.is_sign_negative(), expected.is_sign_negative());
                        }
                    }
                }
            }
            let zero = dyadic_round(&BigInt::ZERO, 1000, p).unwrap();
            assert_eq!(zero.prec(), p);
            assert!(zero.is_zero() && !zero.is_sign_negative());
        }
    }

    #[test]
    fn dyadic_product_chunks_avoid_unbounded_multiplication_algorithms() {
        for (a_bits, b_bits) in [
            (31usize, 89usize),
            (8192, 12001),
            (8193, 17003),
            (20007, 40001),
        ] {
            let a = (BigInt::from(1u32) << a_bits) + 7919u32;
            let b = (BigInt::from(1u32) << b_bits) - 104729u32;
            for (a_sign, b_sign) in [(1i32, 1i32), (-1, 1), (1, -1)] {
                let a = &a * a_sign;
                let b = &b * b_sign;
                assert_eq!(dyadic_multiply(&a, &b), &a * &b);
            }
        }
    }

    #[test]
    fn dyadic_plan_requires_every_legacy_prefix_and_normal_output_range() {
        let matrix = [Float::with_val(128, 3)];
        let basis = [Float::with_val(256, 5)];
        let plan = DyadicFormPlan::new(&matrix, &basis, 1, 128)
            .unwrap()
            .unwrap();
        assert!(plan.normal_range(128, -1000, 1000));
        assert!(!plan.normal_range(128, 1, 1000));
        assert!(!plan.normal_range(128, -1000, 2));
        let tiny = Float::with_val(64, 1) >> (RATIONAL_BIT_BUDGET / 2) as u32;
        assert!(DyadicFormPlan::new(&matrix, &[tiny], 1, 128)
            .unwrap()
            .is_none());
        let nan = [Float::with_val(64, rug::float::Special::Nan)];
        assert!(DyadicFormPlan::new(&nan, &basis, 1, 128).is_err());
    }

    #[test]
    fn dyadic_form_admission_is_ownership_based_and_shared_with_live_parents() {
        let p = 3386;
        let dim = 17;
        let d = 5;
        let matrix = (0..dim * dim)
            .map(|i| Float::with_val(p, i % 17 + 1) >> (i % 67) as u32)
            .collect::<Vec<_>>();
        let basis = (0..dim * d)
            .map(|i| Float::with_val(p + 128, i % 13 + 1) >> (i % 83) as u32)
            .collect::<Vec<_>>();
        let plan = DyadicFormPlan::new(&matrix, &basis, dim, p + 128)
            .unwrap()
            .unwrap();
        let bytes = plan.workspace(dim, d, p + 128, 1).unwrap();
        assert!(plan.workspace(usize::MAX, d, p + 128, 1).is_err());
        let resources = xc_core::RunResources::new(bytes + 4096).unwrap();
        let retained = resources.try_reserve_retained(4097).unwrap();
        assert!(
            dyadic_forms(&matrix, &basis, dim, d, p + 128, &resources, 1)
                .unwrap()
                .is_none()
        );
        assert_eq!(resources.usage().working_bytes, 0);
        drop(retained);
        let expected = reference_forms(&matrix, &basis, dim, d, p + 128).unwrap();
        let actual = dyadic_forms(&matrix, &basis, dim, d, p + 128, &resources, 1)
            .unwrap()
            .unwrap();
        assert_eq!(actual, expected);
        assert_eq!(resources.usage().working_bytes, 0);
    }

    fn full_precision_values(count: usize, p: u32, state: &mut u64) -> Vec<Float> {
        let next = |state: &mut u64| {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            *state
        };
        (0..count)
            .map(|index| {
                if index.is_multiple_of(29) {
                    return Float::with_val(p, 0);
                }
                let mut words = (0..p.div_ceil(32))
                    .map(|_| next(state) as u32)
                    .collect::<Vec<_>>();
                let last = words.len() - 1;
                if !p.is_multiple_of(32) {
                    words[last] &= (1u32 << (p % 32)) - 1;
                }
                words[last] |= 1u32 << ((p - 1) % 32);
                words[0] |= 1;
                let mut value = Integer::from_digits(&words, rug::integer::Order::Lsf);
                if index.is_multiple_of(3) {
                    value = -value;
                }
                let mut value = Float::with_val(p, value);
                value <<= -(p as i32) + (next(state) % 65) as i32 - 32;
                value
            })
            .collect()
    }

    #[test]
    fn dyadic_forms_match_full_significand_reference_at_high_precision() {
        for p in [3386, 6708] {
            let mut seed = 0x91258945199u64;
            let dim = 9;
            let d = 5;
            let matrix = full_precision_values(dim * dim, p, &mut seed);
            let basis = full_precision_values(dim * d, p + 128, &mut seed);
            let expected = reference_forms(&matrix, &basis, dim, d, p + 128).unwrap();
            for workers in [1, 4, 8] {
                let resources =
                    xc_core::RunResources::with_worker_limit(256 << 20, workers).unwrap();
                let actual = dyadic_forms(&matrix, &basis, dim, d, p + 128, &resources, workers)
                    .unwrap()
                    .unwrap();
                for (left, right) in [(&actual.0, &expected.0), (&actual.1, &expected.1)] {
                    for (a, b) in left.iter().zip(right) {
                        assert_eq!(a.prec(), b.prec());
                        assert_eq!(a.as_ord(), b.as_ord());
                        assert_eq!(a.is_sign_negative(), b.is_sign_negative());
                    }
                }
                assert_eq!(resources.usage().working_bytes, 0);
            }
        }
    }

    #[test]
    fn dyadic_fallback_keeps_compact_extreme_source_and_budget_errors() {
        let p = 128;
        let basis = [Float::with_val(p, 1)];
        let resources = xc_core::RunResources::new(4096).unwrap();
        let _scope = resources.enter();
        for (exponent, eligible) in [
            (RATIONAL_BIT_BUDGET / 2, true),
            (RATIONAL_BIT_BUDGET - 64, false),
        ] {
            // A positive exponent contributes numerator bits, not a large
            // denominator. Half the cap is eligible; cap-64 plus this
            // basis's 128-bit lattice exceeds the AV prefix envelope.
            let matrix = [Float::with_val(p, 1) << exponent as u32];
            assert_eq!(
                DyadicFormPlan::new(&matrix, &basis, 1, p)
                    .unwrap()
                    .is_some(),
                eligible
            );
            // Neither compact Float case may allocate its enormous integer
            // under this cap. Both retain the legacy admission-error text.
            let expected = forms_gmp(&matrix, &basis, 1, 1, p, &resources, 1).unwrap_err();
            let actual = forms(&matrix, &basis, 1, 1, p).unwrap_err();
            assert_eq!(actual.to_string(), expected.to_string());
            assert_eq!(resources.usage().working_bytes, 0);
        }
        // Invalid fast-path precision keeps the existing supported legacy
        // behavior rather than adding a new mathematical acceptance condition.
        let matrix = [Float::with_val(p, 2)];
        assert!(DyadicFormPlan::new(&matrix, &basis, 1, 63)
            .unwrap()
            .is_none());
        let expected = forms_gmp(&matrix, &basis, 1, 1, 63, &resources, 1).unwrap();
        assert_eq!(forms(&matrix, &basis, 1, 1, 63).unwrap(), expected);
    }

    #[test]
    fn preparation_worker_override_is_strict_and_capped_without_another_pool() {
        assert_eq!(preparation_worker_limit(122, 122, None).unwrap(), 16);
        assert_eq!(preparation_worker_limit(4, 122, None).unwrap(), 4);
        assert_eq!(preparation_worker_limit(122, 8, Some("32")).unwrap(), 8);
        assert_eq!(preparation_worker_limit(122, 122, Some("1")).unwrap(), 1);
        for invalid in [
            "",
            "0",
            "-1",
            "+1",
            "1.5",
            "1 ",
            "no",
            "184467440737095516160",
        ] {
            assert!(preparation_worker_limit(8, 8, Some(invalid)).is_err());
        }
        assert!(preparation_worker_limit(0, 8, None).is_err());
        assert!(preparation_worker_limit(8, 0, None).is_err());
    }

    #[test]
    fn preparation_map_keeps_order_and_bounds_active_tasks() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        for workers in [1, 3, 8] {
            let active = AtomicUsize::new(0);
            let peak = AtomicUsize::new(0);
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(8)
                .build()
                .unwrap();
            let actual = pool.install(|| {
                preparation_map(43, workers, |index| {
                    let current = active.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(current, Ordering::SeqCst);
                    let value = index * index;
                    active.fetch_sub(1, Ordering::SeqCst);
                    value
                })
            });
            assert_eq!(
                actual,
                (0..43).map(|index| index * index).collect::<Vec<_>>()
            );
            assert!(peak.load(Ordering::SeqCst) <= workers);
            assert_eq!(active.load(Ordering::SeqCst), 0);
        }
    }

    #[test]
    fn streamed_forms_keep_high_precision_zero_and_cancellation_values() {
        for p in [128, 3386, 6708] {
            let dim = 9;
            let d = 5;
            let matrix = (0..dim * dim)
                .map(|index| Float::with_val(p, (index as i32 % 11) - 5) >> (index % 37) as u32)
                .collect::<Vec<_>>();
            let v = (0..dim * d)
                .map(|index| {
                    Float::with_val(p + 128, (index as i32 % 7) - 3) >> (index % 101) as u32
                })
                .collect::<Vec<_>>();
            let expected = reference_forms(&matrix, &v, dim, d, p + 128).unwrap();
            for workers in [1, 4, 16] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(workers)
                    .build()
                    .unwrap();
                let actual = pool
                    .install(|| forms(&matrix, &v, dim, d, p + 128))
                    .unwrap();
                for (left, right) in [(&actual.0, &expected.0), (&actual.1, &expected.1)] {
                    for (a, b) in left.iter().zip(right) {
                        assert_eq!(a.prec(), b.prec());
                        assert_eq!(a.as_ord(), b.as_ord());
                        assert_eq!(a.is_sign_negative(), b.is_sign_negative());
                    }
                }
            }
        }
    }

    #[test]
    fn arithmetic_shapes_admit_resource_bounded_n800_without_dimension_caps() {
        shape(800, 3514).unwrap();
        form_shape(1601, 65, 1601 * 1601, 1601 * 65).unwrap();
        assert!(shape(usize::MAX, 128).is_err());
        assert!(shape(0, 128).is_err());
        assert!(form_shape(usize::MAX, 65, 1, 1).is_err());
        assert!(form_shape(1601, 65, 1601 * 1601 - 1, 1601 * 65).is_err());
        let lattice = lattice(799, 800);
        assert_eq!(lattice, Rational::from((638_401, 640_000)));
    }

    #[test]
    fn exact_form_reservation_uses_remaining_capacity_and_releases_on_return() {
        let p = 128;
        let matrix = [1, 0, 0, 1].map(|value| Float::with_val(p, value));
        let basis = [1, 1].map(|value| Float::with_val(p, value));
        let expected = reference_forms(&matrix, &basis, 2, 1, p).unwrap();
        let resources = xc_core::RunResources::new(4096).unwrap();
        let _scope = resources.enter();
        let retained = resources.try_reserve_retained(4095).unwrap();
        let error = forms(&matrix, &basis, 2, 1, p).unwrap_err();
        assert!(error.to_string().contains("remaining workspace"));
        assert_eq!(resources.usage().working_bytes, 0);
        drop(retained);
        let actual = forms(&matrix, &basis, 2, 1, p).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(resources.usage().working_bytes, 0);
        assert!(resources.usage().peak_bytes > 0);
    }

    #[test]
    fn exact_square_reserves_actual_exponent_span_and_preserves_its_point() {
        let p = 128;
        let value = Float::with_val(p, 1) >> 10_000u32;
        let resources = xc_core::RunResources::new(1 << 20).unwrap();
        let _scope = resources.enter();
        let retained = resources.try_reserve_retained((1 << 20) - 1).unwrap();
        let error = square_sum(std::slice::from_ref(&value), p).unwrap_err();
        assert!(error
            .to_string()
            .contains("exact square exceeds remaining workspace"));
        assert_eq!(resources.usage().working_bytes, 0);
        drop(retained);
        let actual = square_sum(std::slice::from_ref(&value), p).unwrap();
        assert_eq!(actual, Float::with_val(p, 1) >> 20_000u32);
        assert_eq!(actual.prec(), p);
        assert_eq!(resources.usage().working_bytes, 0);
    }

    #[test]
    fn parallel_basis_is_bit_identical_to_serial_reference_at_any_thread_count() {
        for (n, d, count, p) in [(6, 4, 40, 192), (9, 3, 12, 128), (5, 6, 23, 256)] {
            let l = Float::with_val(p, 3) + Float::with_val(p, 1) / 7u32;
            let mut z = (0..count)
                .map(|j| Float::with_val(p, 2 * j as u32 + 1) / (3 * count as u32))
                .collect::<Vec<_>>();
            if count == 12 {
                // A prefix root on a lattice point (k/n)^2 zeroes that mode.
                z[1] = Float::with_val(p, Rational::from((4, 81)));
            }
            let expected = reference_basis(&z, &l, n, d, p);
            for threads in [1, 2, 4, 8] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                let actual = pool.install(|| basis(&z, &l, n, d, p));
                match (&actual, &expected) {
                    (Ok(a), Ok(b)) => {
                        assert_eq!(a.len(), b.len());
                        for (x, y) in a.iter().zip(b) {
                            assert!(x.prec() == y.prec() && x.as_ord() == y.as_ord());
                        }
                    }
                    (Err(a), Err(b)) => assert_eq!(a.to_string(), b.to_string()),
                    _ => panic!("parallel and serial bases disagree on failure"),
                }
            }
        }
    }

    /// Serial `basis` retained verbatim from before mode parallelism.
    fn reference_basis(z: &[Float], l: &Float, n: usize, d: usize, p: u32) -> Result<Vec<Float>> {
        shape(n, p)?;
        ensure!(
            (1..=65).contains(&d) && d <= n + 1 && z.len() >= n + 1 - d && l > &0,
            "invalid preparation basis shape"
        );
        let prefix = n + 1 - d;
        let roots = z[..prefix].iter().map(source).collect::<Result<Vec<_>>>()?;
        let length = source(l)?;
        let mut facts = vec![Integer::from(1); 2 * n + 1];
        for k in 1..=2 * n {
            facts[k] = facts[k - 1].clone() * k;
        }
        let power = Integer::from(n).pow(2 * n as u32);
        let mut v = vec![Float::with_val(p, 0); (2 * n + 1) * d];
        for k in 0..=n {
            let a = lattice(k, n);
            let mut q = Rational::from((power.clone(), facts[n - k].clone() * &facts[n + k]));
            if k % 2 == 1 {
                q = -q;
            }
            for r in &roots {
                q *= a.clone() - r;
                check(&q)?;
            }
            for j in 0..d {
                let mut value = None;
                for guard in [64, 128, 256, 512, 1024, 2048, 4096] {
                    let w = p + guard;
                    let r = I::from_rational(&q, w).div(&I::from_rational(&length, w).sqrt()?)?;
                    if let Some(r) = rounded(&r, p)? {
                        value = Some(r);
                        break;
                    }
                }
                let value = value
                    .ok_or_else(|| anyhow::anyhow!("preparation basis rounding unresolved"))?;
                v[(n + k) * d + j] = value.clone();
                v[(n - k) * d + j] = value;
                q *= &a;
                check(&q)?;
            }
        }
        Ok(v)
    }

    #[test]
    fn parallel_head_is_bit_identical_to_serial_reference_at_any_thread_count() {
        for (n, d, count, p) in [(6, 4, 40, 192), (3, 4, 9, 128), (5, 2, 23, 256)] {
            let l = Float::with_val(p, 3) + Float::with_val(p, 1) / 7u32;
            let t = (0..count)
                .map(|j| Float::with_val(p, 1 + 3 * j as u32) / 5u32)
                .collect::<Vec<_>>();
            // Coordinates avoid the lattice points (k/n)^2, except one pole case.
            let mut z = (0..count)
                .map(|j| Float::with_val(p, 2 * j as u32 + 1) / (3 * count as u32))
                .collect::<Vec<_>>();
            if count == 9 {
                z[7] = Float::with_val(p, Rational::from((4, 9)));
            }
            let expected = reference_head(&t, &z, &l, n, d, p);
            for threads in [1, 2, 4, 8] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                for _ in 0..2 {
                    let actual = pool.install(|| head(&t, &z, &l, n, d, p));
                    match (&actual, &expected) {
                        (Ok(a), Ok(b)) => {
                            assert_eq!(a.len(), b.len());
                            for (x, y) in a.iter().zip(b) {
                                assert!(x.prec() == y.prec() && x.as_ord() == y.as_ord());
                            }
                        }
                        (Err(a), Err(b)) => assert_eq!(a.to_string(), b.to_string()),
                        _ => panic!("parallel and serial heads disagree on failure"),
                    }
                }
            }
        }
    }

    /// Serial `head` retained verbatim from before ordinate parallelism.
    fn reference_head(
        t: &[Float],
        z: &[Float],
        l: &Float,
        n: usize,
        d: usize,
        p: u32,
    ) -> Result<Vec<Float>> {
        shape(n, p)?;
        ensure!(
            d > 0
                && d <= 65
                && d <= n + 1
                && t.len() == z.len()
                && t.len() >= n + 1 - d
                && t.len() <= 1_000_000,
            "invalid preparation head shape"
        );
        let prefix = n + 1 - d;
        let l = source(l)?;
        crate::ccm::certified_roots::boundary::rational_budget(z.iter(), 1)?;
        let z = z.iter().map(source).collect::<Result<Vec<_>>>()?;
        // Stream exact coefficients: their aggregate size can exceed the budget
        // even though a single coefficient and the moment accumulators fit. Keep
        // the same exact factors and directed summation order at each guard tier.
        for guard in [64, 128, 256, 512, 1024, 2048, 4096] {
            let w = p + guard;
            let mut moments = vec![I::from_i64(0, w); 2 * d - 1];
            for (j, t) in t.iter().enumerate().skip(prefix) {
                let theta = source(t)? * &l / 2;
                let mut factor = Rational::from(1);
                for r in &z[..prefix] {
                    factor *= z[j].clone() - r;
                    check(&factor)?;
                }
                for k in 1..=n {
                    let denominator = z[j].clone() - lattice(k, n);
                    ensure!(
                        denominator != 0,
                        "stored preparation coordinate is a rational pole"
                    );
                    factor /= denominator;
                    check(&factor)?;
                }
                let factor = factor.square();
                check(&factor)?;
                let theta_i = I::from_rational(&theta, w);
                let sinc = if theta == 0 {
                    I::from_i64(1, w)
                } else {
                    theta_i.sin().div(&theta_i)?
                };
                let mut weight = sinc.square().mul(&I::from_rational(&factor, w));
                let z = I::from_rational(&z[j], w);
                for m in &mut moments {
                    *m = m.add(&weight);
                    weight = weight.mul(&z);
                }
            }
            let values = moments
                .iter()
                .map(|x| rounded(x, p))
                .collect::<Result<Option<Vec<_>>>>()?;
            if let Some(values) = values {
                return Ok((0..d * d).map(|k| values[k / d + k % d].clone()).collect());
            }
        }
        bail!("preparation head rounding unresolved within guard budget")
    }

    #[test]
    fn parallel_forms_are_bit_identical_to_serial_reference_at_any_thread_count() {
        for (dim, d, p) in [(9, 5, 256), (13, 7, 512), (3, 1, 64)] {
            let value = |seed: usize, scale: u32| {
                let numerator = (seed * 7919 + 104_729) % 65_521;
                let sign = if seed.is_multiple_of(3) { -1 } else { 1 };
                Float::with_val(p, sign * numerator as i64) >> (scale + (seed % 23) as u32)
            };
            let matrix = (0..dim * dim).map(|k| value(k, 8)).collect::<Vec<_>>();
            let v = (0..dim * d).map(|k| value(k + 7, 4)).collect::<Vec<_>>();
            let expected = reference_forms(&matrix, &v, dim, d, p).unwrap();
            for threads in [1, 2, 4, 8] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                for _ in 0..2 {
                    let actual = pool.install(|| forms(&matrix, &v, dim, d, p)).unwrap();
                    for (left, right) in [(&actual.0, &expected.0), (&actual.1, &expected.1)] {
                        assert_eq!(left.len(), right.len());
                        for (a, b) in left.iter().zip(right) {
                            assert!(a.prec() == b.prec() && a.as_ord() == b.as_ord());
                        }
                    }
                }
            }
        }
    }

    /// Serial `forms` retained verbatim from before row parallelism.
    fn reference_forms(
        matrix: &[Float],
        v: &[Float],
        dim: usize,
        d: usize,
        p: u32,
    ) -> Result<(Vec<Float>, Vec<Float>)> {
        ensure!(
            dim <= 1025
                && dim > 0
                && d > 0
                && d <= 65
                && matrix.len() == dim * dim
                && v.len() == dim * d,
            "preparation form shape mismatch"
        );
        let estimate = matrix
            .iter()
            .chain(v)
            .try_fold(0u128, |sum, x| -> Result<_> {
                ensure!(x.is_finite(), "nonfinite preparation form source");
                Ok(sum
                    + u128::from(x.prec())
                    + u128::from(i64::from(x.get_exp().unwrap_or(0)).unsigned_abs())
                    + 2)
            })?;
        // The estimate counts bits; the declared working budget counts bytes.
        let budget_bits = u128::from(
            crate::ccm::capture_runtime::CaptureResourcePolicy::from_environment()?
                .maximum_working_bytes,
        ) * 8;
        ensure!(
            estimate * (d as u128 + 4) <= budget_bits,
            "preparation exact form exceeds the declared workspace budget"
        );
        let v = v.iter().map(source).collect::<Result<Vec<_>>>()?;
        let mut av = vec![Rational::new(); dim * d];
        for r in 0..dim {
            for k in 0..dim {
                let a: Rational =
                    (source(&matrix[r * dim + k])? + source(&matrix[k * dim + r])?) / 2;
                for j in 0..d {
                    av[r * d + j] += a.clone() * &v[k * d + j];
                    check(&av[r * d + j])?;
                }
            }
        }
        let mut gram = vec![Float::with_val(p, 0); d * d];
        let mut actual = gram.clone();
        for i in 0..d {
            for j in 0..=i {
                let mut g = Rational::new();
                let mut a = Rational::new();
                for k in 0..dim {
                    g += v[k * d + i].clone() * &v[k * d + j];
                    a += v[k * d + i].clone() * &av[k * d + j];
                    check(&g)?;
                    check(&a)?;
                }
                let g = round(&g, p)?;
                let a = round(&(a / 2), p)?;
                gram[i * d + j] = g.clone();
                gram[j * d + i] = g;
                actual[i * d + j] = a.clone();
                actual[j * d + i] = a;
            }
        }
        Ok((gram, actual))
    }

    #[test]
    fn preparation_basis_matches_closed_rational_coefficients() {
        for p in [64, 128, 256] {
            let v = basis(&[], &Float::with_val(p, 4), 2, 3, p).unwrap();
            let coefficient = [
                Rational::from((1, 3)),
                Rational::from((-4, 3)),
                Rational::from(2),
                Rational::from((-4, 3)),
                Rational::from((1, 3)),
            ];
            for row in 0..5 {
                let coordinate = Rational::from(((row as i32 - 2).pow(2), 4));
                for j in 0..3 {
                    assert_eq!(
                        v[row * 3 + j],
                        Float::with_val(
                            p,
                            coefficient[row].clone() * coordinate.clone().pow(j as u32)
                        )
                    );
                }
            }
        }
    }
    #[test]
    fn preparation_forms_match_exact_closed_quadratic_values() {
        for p in [64, 128, 256] {
            for shift in [0u32, 200, 600] {
                let a = Float::with_val(p, 1) << shift;
                let v = vec![
                    Float::with_val(p, 1),
                    a.clone(),
                    Float::with_val(p, 2),
                    Float::with_val(p, 1),
                    Float::with_val(p, 3),
                    -a.clone(),
                ];
                let matrix = [2, 7, 0, -7, 4, 5, 0, -5, 6].map(|x| Float::with_val(p, x));
                let (g, f) = forms(&matrix, &v, 3, 2, p).unwrap();
                let a = a.to_rational().unwrap();
                let want_g = [
                    Rational::from(14),
                    a.clone() * -2 + 2,
                    a.clone() * -2 + 2,
                    a.clone().square() * 2 + 1,
                ];
                let want_f = [
                    Rational::from(36),
                    a.clone() * -8 + 4,
                    a.clone() * -8 + 4,
                    a.square() * 4 + 2,
                ];
                for k in 0..4 {
                    assert_eq!(g[k], Float::with_val(p, &want_g[k]));
                    assert_eq!(f[k], Float::with_val(p, &want_f[k]));
                }
            }
        }
    }
    #[test]
    fn preparation_head_matches_independent_high_precision_formula() {
        for p in [64, 128, 256] {
            for n in [1, 2] {
                let l = Float::with_val(p, 4);
                let t = [1, 2, 3].map(|x| Float::with_val(p, x));
                let z = [0.125, 0.375, 0.75].map(|x| Float::with_val(p, x));
                let got = head(&t, &z, &l, n, 2, p).unwrap();
                let mut moments = vec![Float::with_val(2048, 0); 3];
                for j in n - 1..3 {
                    let theta = Float::with_val(2048, &t[j]) * 2u32;
                    let mut f = theta.clone().sin() / theta;
                    if n == 2 {
                        f *= Float::with_val(2048, &z[j]) - &z[0];
                    }
                    for k in 1..=n {
                        f /= Float::with_val(2048, &z[j])
                            - Float::with_val(2048, Rational::from((k * k, n * n)));
                    }
                    let mut weight = f.square();
                    for m in &mut moments {
                        *m += &weight;
                        weight *= &z[j];
                    }
                }
                for k in 0..4 {
                    assert_eq!(got[k], Float::with_val(p, &moments[k / 2 + k % 2]));
                }
            }
        }
    }
    #[test]
    fn preparation_head_streams_coefficients_beyond_aggregate_storage_budget() {
        // Each exact squared factor uses about 4 million bits; retaining all
        // twenty exceeded the 64-Mibit coefficient budget. Streaming needs one.
        // H_k=20*z^k/(1-z)^2 rounds to 20*z^k at 64 bits for this tiny z.
        let p = 64;
        let z = Float::with_val(p, 1) >> 1_000_000u32;
        let head = head(
            &vec![Float::with_val(p, 0); 20],
            &vec![z.clone(); 20],
            &Float::with_val(p, 1),
            1,
            2,
            p,
        )
        .unwrap();
        for (k, value) in head.iter().enumerate() {
            assert_eq!(
                *value,
                Float::with_val(p, 20) * z.clone().pow((k / 2 + k % 2) as u32)
            );
        }
    }
    #[test]
    fn preparation_tail_split_handles_cancellation_and_rejects_outside_domain() {
        let resources = xc_core::RunResources::new(16 << 20).unwrap();
        for p in [64, 128, 256] {
            let a = Float::with_val(p, 1);
            let h = Float::with_val(p, 1) << 400u32;
            let (target, t, _output) = test_split(
                std::slice::from_ref(&a),
                std::slice::from_ref(&h),
                p,
                &resources,
            )
            .unwrap();
            assert!(target >= 400);
            assert_eq!(
                a.to_rational().unwrap(),
                h.to_rational().unwrap() + t[0].to_rational().unwrap()
            );
        }
        assert!(coordinates(&[Float::with_val(64, 1)], &Float::with_val(64, 1), 0, 64).is_err());
        assert!(basis(&[], &Float::with_val(64, 1), 513, 1, 64).is_err());
        assert!(head(
            &[Float::with_val(64, 1)],
            &[Float::with_val(64, 1)],
            &Float::with_val(64, 1),
            1,
            2,
            64
        )
        .is_err());
        assert!(square_sum(&[Float::with_val(64, rug::float::Special::Nan)], 64).is_err());
    }

    fn test_split(
        actual: &[Float],
        head: &[Float],
        p: u32,
        resources: &xc_core::RunResources,
    ) -> Result<(u32, Vec<Float>, xc_core::MemoryReservation)> {
        split(actual, head, p, resources, |target| {
            let bytes = actual.len() as u64 * (u64::from(target).div_ceil(64) * 8 + 96);
            Ok(resources.try_reserve_working(bytes)?)
        })
    }

    // The original all-Rational split, retained as an ordered reference.
    fn reference_split(actual: &[Float], head: &[Float], p: u32) -> Result<(u32, Vec<Float>)> {
        ensure!(
            actual.len() == head.len(),
            "preparation tail shape mismatch"
        );
        let tail = actual
            .iter()
            .zip(head)
            .map(|(a, h)| Ok(source(a)? - source(h)?))
            .collect::<Result<Vec<_>>>()?;
        let target = tail
            .iter()
            .map(|q| q.numer().significant_bits())
            .max()
            .unwrap_or(p)
            .max(p);
        ensure!(
            target <= 1_000_000,
            "exact preparation tail exceeds representation precision budget"
        );
        let values = tail
            .iter()
            .map(|q| {
                let value = round(q, target)?;
                ensure!(
                    value.to_rational().as_ref() == Some(q),
                    "tail split lost exact dyadic difference"
                );
                Ok(value)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok((target, values))
    }

    #[test]
    fn streamed_split_matches_original_full_precision_values_and_target() {
        for p in [128, 3386] {
            let mut seed = 0x91258945199;
            let actual = full_precision_values(37, p, &mut seed);
            let head = full_precision_values(37, p, &mut seed);
            let expected = reference_split(&actual, &head, p).unwrap();
            let resources = xc_core::RunResources::new(64 << 20).unwrap();
            let (target, values, output) = test_split(&actual, &head, p, &resources).unwrap();
            assert_eq!(target, expected.0);
            for (a, b) in values.iter().zip(&expected.1) {
                assert_eq!(a.prec(), b.prec());
                assert_eq!(a.as_ord(), b.as_ord());
                assert_eq!(a.is_sign_negative(), b.is_sign_negative());
            }
            assert_eq!(resources.usage().working_bytes, output.bytes());
            drop((values, output));
            assert_eq!(resources.usage().working_bytes, 0);
        }
    }

    #[test]
    fn streamed_split_keeps_later_source_error_before_global_precision_refusal() {
        let p = 128;
        let huge = Float::with_val(p, 1) << 1_000_001u32;
        let nan = Float::with_val(p, rug::float::Special::Nan);
        let actual = [huge, nan];
        let head = [Float::with_val(p, 1), Float::with_val(p, 0)];
        let resources = xc_core::RunResources::new(64 << 20).unwrap();
        let expected = reference_split(&actual, &head, p).unwrap_err();
        let error = split(&actual, &head, p, &resources, |_| {
            panic!("output cannot be admitted after invalid sources")
        })
        .unwrap_err();
        assert_eq!(error.to_string(), expected.to_string());
        assert_eq!(error.to_string(), "nonfinite exact source point");
        let expected = reference_split(&actual[..1], &head[..1], p).unwrap_err();
        let error = split(&actual[..1], &head[..1], p, &resources, |_| {
            panic!("output cannot be admitted above the precision limit")
        })
        .unwrap_err();
        assert_eq!(error.to_string(), expected.to_string());
        assert_eq!(resources.usage().working_bytes, 0);
    }

    #[test]
    fn streamed_split_admits_scratch_and_promoted_output_before_work() {
        let actual = [Float::with_val(128, 1)];
        let head = [Float::with_val(128, 1) << 400u32];
        let bytes = split_scalar_bytes(&actual, &head).unwrap();
        let resources = xc_core::RunResources::new(bytes + 1024).unwrap();
        let retained = resources.try_reserve_retained(1025).unwrap();
        let error = split(&actual, &head, 128, &resources, |_| {
            panic!("output callback reached without scratch admission")
        })
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("exact split exceeds remaining workspace"));
        assert_eq!(resources.usage().working_bytes, 0);
        drop(retained);
        let called = std::cell::Cell::new(false);
        let error = split(&actual, &head, 128, &resources, |target| {
            called.set(true);
            assert!(target >= 400);
            assert_eq!(resources.usage().working_bytes, bytes);
            Ok(resources.try_reserve_working(2048)?)
        })
        .unwrap_err();
        assert!(called.get());
        assert!(error.to_string().contains("run memory budget exceeded"));
        assert_eq!(resources.usage().working_bytes, 0);
    }
}
