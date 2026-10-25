//! Correct rounding of response dot products and Euclidean/Frobenius norms.
//! Exact stored points only; this does not bound source or linear-solve error.
use super::root_response_math::GuardPolicy;
use crate::ccm::retained_evidence::finite_math::{scale, scale_float};
use anyhow::{bail, Result};
use rayon::prelude::*;
use rug::{float::Round, Float};
use xc_numerics::mpfr_interval::MpfrInterval as I;
const GUARDS: [u32; 7] = [64, 128, 256, 512, 1024, 2048, 4096];

// The established prefix keeps its existing admission. An extended attempt
// owns its additional scratch until all interval temporaries have dropped.
// Sources are already owned by the caller and are not charged twice here.
fn reserve_extended_attempt(guard: u32, bytes: u128) -> Result<Option<xc_core::MemoryReservation>> {
    if guard <= 4096 {
        return Ok(None);
    }
    reserve_point_workspace(bytes)
}
fn reserve_point_workspace(bytes: u128) -> Result<Option<xc_core::MemoryReservation>> {
    let bytes = u64::try_from(bytes).map_err(|_| {
        anyhow::anyhow!("run memory budget exceeded: response point workspace overflow")
    })?;
    if let Some(resources) = xc_core::RunResources::current() {
        // Never wait for an owner that may already hold prepared sources.
        Ok(Some(resources.try_reserve_working(bytes)?))
    } else if u128::from(bytes) > super::source_working_budget()? {
        bail!("run memory budget exceeded: response point workspace requested={bytes}")
    } else {
        Ok(None)
    }
}
fn vector_workspace_bytes(count: usize, work: u32) -> u128 {
    (count as u128 * 32 + 128) * (u128::from(work).div_ceil(8) + 96)
}
fn dense_scratch_bytes(entries: usize, vectors: usize, work: u32, floats: u128) -> u128 {
    (entries as u128 * floats + vectors as u128 * 32 + 128) * (u128::from(work).div_ceil(8) + 96)
}

// Bound both the product storage and a full exponent-span accumulator before
// asking MPFR to sum. This deliberately overcharges its sparse summation
// scratch, including pointers and simultaneous normalized scalar operands.
fn exact_dot_workspace(left: &[Float], right: &[Float], le: i64, re: i64, p: u32) -> Result<u128> {
    let mut product_bytes = 0u128;
    let mut largest = 0u128;
    let mut low = 0i64;
    let mut high = 0i64;
    for (a, b) in left.iter().zip(right) {
        let bits = a
            .prec()
            .checked_add(b.prec())
            .ok_or_else(|| anyhow::anyhow!("response exact product precision overflow"))?;
        let bytes = u128::from(bits).div_ceil(8) + 96;
        product_bytes += bytes;
        largest = largest.max(bytes);
        if let (Some(ae), Some(be)) = (a.get_exp(), b.get_exp()) {
            let upper = i64::from(ae) - le + i64::from(be) - re;
            high = high.max(upper);
            low = low.min(upper - i64::from(bits));
        }
    }
    let carry = u64::from(usize::BITS - left.len().max(1).leading_zeros());
    let span = u128::try_from(high - low)? + u128::from(carry) + u128::from(p) + 2;
    Ok((product_bytes + largest * 4 + span.div_ceil(8) * 4 + left.len() as u128 * 32 + 512) * 4)
}

fn exact_dot(left: &[Float], right: &[Float], p: u32, le: i64, re: i64) -> Result<Float> {
    let _reservation = reserve_point_workspace(exact_dot_workspace(left, right, le, re, p)?)?;
    let mut products = Vec::with_capacity(left.len());
    for (a, b) in left.iter().zip(right) {
        let bits = a
            .prec()
            .checked_add(b.prec())
            .ok_or_else(|| anyhow::anyhow!("response exact product precision overflow"))?;
        let a = scale_float(a, -le, a.prec())?;
        let b = scale_float(b, -re, b.prec())?;
        let (product, direction) = Float::with_val_round(bits, &a * &b, Round::Nearest);
        if !product.is_finite()
            || direction != std::cmp::Ordering::Equal
            || (product.is_zero() && !a.is_zero() && !b.is_zero())
        {
            bail!("response exact dot product exceeds the intermediate range");
        }
        products.push(product);
    }
    let (value, direction) = Float::with_val_round(p, Float::sum(products.iter()), Round::Nearest);
    if !value.is_finite() || (value.is_zero() && direction != std::cmp::Ordering::Equal) {
        bail!("response exact dot product exceeds the output range");
    }
    scale_float(&value, le + re, p)
}

fn precision<'a>(values: impl Iterator<Item = &'a Float>, p: u32, count: usize) -> Result<u32> {
    let working = source_precision(values, p)?;
    point_workspace(working, count)?;
    Ok(working)
}
fn source_precision<'a>(values: impl Iterator<Item = &'a Float>, p: u32) -> Result<u32> {
    if !(64..=1_000_000).contains(&p) {
        bail!("unsupported response point precision");
    }
    let mut working = p;
    for value in values {
        if !value.is_finite() || value.prec() > 1_000_128 {
            bail!("response point source is nonfinite or exceeds supported precision");
        }
        working = working.max(value.prec());
    }
    Ok(working)
}
fn point_workspace(working: u32, count: usize) -> Result<()> {
    if count as u128 * (u128::from(working + 4096).div_ceil(8) + 96) * 32
        > super::source_working_budget()?
    {
        bail!("response point arithmetic exceeds the workspace budget");
    }
    Ok(())
}
// Dense shifted entries, their squared intervals and the scaled sum's
// aligned intervals are the three simultaneous interval arrays in the
// bordered path. Source points remain live; vector/scalar work is separate.
fn dense_workspace_bytes(
    entries: usize,
    vectors: usize,
    source_precision: u32,
    work: u32,
    working_floats: u128,
) -> u128 {
    let source_point = u128::from(source_precision).div_ceil(8) + 96;
    let work_point = u128::from(work).div_ceil(8) + 96;
    entries as u128 * source_point
        + (entries as u128 * working_floats + vectors as u128 * 32 + 128) * work_point
}
fn dense_workspace(
    entries: usize,
    vectors: usize,
    source_precision: u32,
    work: u32,
    working_floats: u128,
) -> Result<()> {
    if dense_workspace_bytes(entries, vectors, source_precision, work, working_floats)
        > super::source_working_budget()?
    {
        bail!(
            "response dense point arithmetic exceeds the declared workspace at {work} working bits"
        );
    }
    Ok(())
}
fn sum(values: &[I], p: u32) -> Result<I> {
    for value in values {
        value.validate()?;
    }
    Ok(I::new(
        Float::with_val_round(p, Float::sum(values.iter().map(I::lower)), Round::Down).0,
        Float::with_val_round(p, Float::sum(values.iter().map(I::upper)), Round::Up).0,
    )?)
}
fn rounded(value: &I, p: u32) -> Result<Option<Float>> {
    value.validate()?;
    let lo = Float::with_val(p, value.lower());
    let hi = Float::with_val(p, value.upper());
    if lo == hi
        && lo.is_finite()
        && (!lo.is_zero() || (value.lower().is_zero() && value.upper().is_zero()))
    {
        Ok(Some(lo))
    } else {
        Ok(None)
    }
}
fn norm_enclosure(values: &[I], p: u32) -> Result<I> {
    for value in values {
        value.validate()?;
    }
    let exponent = values
        .iter()
        .flat_map(|v| [v.lower(), v.upper()])
        .filter_map(Float::get_exp)
        .max()
        .map_or(0, i64::from);
    let squares = values
        .iter()
        .map(|value| scale(value, -exponent).map(|x| x.square()))
        .collect::<Result<Vec<_>>>()?;
    scale(&sum(&squares, p)?.sqrt()?, exponent)
}
pub(super) fn dot(left: &[Float], right: &[Float], p: u32) -> Result<Float> {
    if left.len() != right.len() || left.len() > 16385 {
        bail!("response dot product shape is invalid");
    }
    let base = precision(left.iter().chain(right), p, left.len())?;
    let le = left
        .iter()
        .filter_map(Float::get_exp)
        .max()
        .map_or(0, i64::from);
    let re = right
        .iter()
        .filter_map(Float::get_exp)
        .max()
        .map_or(0, i64::from);
    if let Some(value) = interval_dot(left, right, p, base, le, re)? {
        return Ok(value);
    }
    // Multiplying stored binary points at the sum of their precisions is exact.
    // Round the complete exact product sum only once; interval widening cannot
    // otherwise settle extreme cancellation or a rounding-boundary equality.
    exact_dot(left, right, p, le, re)
}
fn interval_dot(
    left: &[Float],
    right: &[Float],
    p: u32,
    base: u32,
    le: i64,
    re: i64,
) -> Result<Option<Float>> {
    for guard in GUARDS {
        let work = base + guard;
        let products = left
            .iter()
            .zip(right)
            .map(|(a, b)| -> Result<I> {
                Ok(I::point(scale_float(a, -le, work)?).mul(&I::point(scale_float(b, -re, work)?)))
            })
            .collect::<Result<Vec<_>>>()?;
        let value = scale(&sum(&products, work)?, le + re)?;
        if let Some(value) = rounded(&value, p)? {
            return Ok(Some(value));
        }
    }
    Ok(None)
}
pub(super) fn norm(values: &[Float], p: u32) -> Result<Float> {
    if values.len() > 16385 {
        bail!("response vector norm shape exceeds the supported limit");
    }
    let base = precision(values.iter(), p, values.len())?;
    let policy = GuardPolicy::configured(p)?;
    for guard in policy.guards() {
        let work = base + guard;
        let _reservation =
            reserve_extended_attempt(guard, vector_workspace_bytes(values.len(), work))?;
        let points = values
            .iter()
            .map(|v| I::point(Float::with_val(work, v)))
            .collect::<Vec<_>>();
        let value = norm_enclosure(&points, work)?;
        if let Some(value) = rounded(&value, p)? {
            return Ok(value);
        }
    }
    bail!(
        "response norm rounding unresolved within {} guard bits",
        policy.maximum_guard_bits()
    )
}
pub(super) fn shifted_norm(
    matrix: &[Float],
    eigenvalue: &Float,
    dimension: usize,
    p: u32,
) -> Result<Float> {
    if dimension == 0 || dimension > 16385 || dimension.checked_mul(dimension) != Some(matrix.len())
    {
        bail!("shifted response norm requires a nonempty square matrix");
    }
    let base = precision(
        matrix.iter().chain(std::iter::once(eigenvalue)),
        p,
        dimension + 1,
    )?;
    let policy = GuardPolicy::configured(p)?;
    for guard in policy.guards() {
        let work = base + guard;
        dense_workspace(matrix.len(), dimension + 1, base, work, 4)?;
        let _reservation = reserve_extended_attempt(
            guard,
            dense_scratch_bytes(matrix.len(), dimension + 1, work, 4),
        )?;
        let eigenvalue = I::point(Float::with_val(work, eigenvalue));
        // Subtract the common diagonal before choosing the norm's scale so a
        // huge common shift cannot erase tiny off-diagonal feedback.
        let shifted = matrix
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let value = I::point(Float::with_val(work, value));
                if index / dimension == index % dimension {
                    value.sub(&eigenvalue)
                } else {
                    value
                }
            })
            .collect::<Vec<_>>();
        let value = norm_enclosure(&shifted, work)?;
        if let Some(value) = rounded(&value, p)? {
            return Ok(value);
        }
    }
    bail!(
        "shifted response norm rounding unresolved within {} guard bits",
        policy.maximum_guard_bits()
    )
}

// Keep binary exponents separate so residual and denominator products need not
// fit in MPFR's exponent range. Every mantissa operation remains directed.
struct Scaled {
    value: I,
    exponent: i64,
}
impl Scaled {
    fn new(value: I, exponent: i64) -> Result<Self> {
        value.validate()?;
        let shift = [value.lower(), value.upper()]
            .into_iter()
            .filter_map(Float::get_exp)
            .max()
            .map_or(0, i64::from);
        Ok(Self {
            value: scale(&value, -shift)?,
            exponent: exponent + shift,
        })
    }
    fn point(value: &Float, p: u32) -> Result<Self> {
        let exponent = value.get_exp().map_or(0, i64::from);
        Ok(Self {
            value: I::point(scale_float(value, -exponent, p)?),
            exponent,
        })
    }
    fn zero(&self) -> bool {
        self.value.lower().is_zero() && self.value.upper().is_zero()
    }
    fn neg(&self) -> Self {
        Self {
            value: self.value.neg(),
            exponent: self.exponent,
        }
    }
    fn mul(&self, other: &Self) -> Result<Self> {
        Self::new(self.value.mul(&other.value), self.exponent + other.exponent)
    }
    fn div(&self, other: &Self) -> Result<Self> {
        Self::new(
            self.value.div(&other.value)?,
            self.exponent - other.exponent,
        )
    }
    fn sqrt(&self) -> Result<Self> {
        let half = self.exponent.div_euclid(2);
        Self::new(scale(&self.value, self.exponent - 2 * half)?.sqrt()?, half)
    }
    fn sum(values: &[Self], p: u32) -> Result<Self> {
        let exponent = values
            .iter()
            .filter(|v| !v.zero())
            .map(|v| v.exponent)
            .max()
            .unwrap_or(0);
        let parts = values
            .iter()
            .map(|v| scale(&v.value, v.exponent - exponent))
            .collect::<Result<Vec<_>>>()?;
        Self::new(sum(&parts, p)?, exponent)
    }
    fn norm(values: &[Self], p: u32) -> Result<Self> {
        let squares = values
            .iter()
            .map(|v| Self::new(v.value.square(), 2 * v.exponent))
            .collect::<Result<Vec<_>>>()?;
        let sum = Self::sum(&squares, p)?;
        let half = sum.exponent.div_euclid(2);
        Self::new(scale(&sum.value, sum.exponent - 2 * half)?.sqrt()?, half)
    }
    fn materialize(&self) -> Result<I> {
        scale(&self.value, self.exponent)
    }
}

// One immutable source binding and at most one actual working precision. The
// borrows prevent source mutation/replacement; neither rounded source keys nor
// caller-supplied norms are accepted as identities. Dropping/replacing this slot
// releases its reservation only after the retained interval allocations drop.
pub(super) struct BorderedResidualSource<'a> {
    matrix: &'a [Float],
    eigenvalue: &'a Float,
    p: u32,
    source_precision: Option<u32>,
    prepared: Option<PreparedBorderedResidual>,
}
struct PreparedBorderedResidual {
    work: u32,
    exponent_range: (i32, i32),
    shifted: Vec<Scaled>,
    matrix_norm: Scaled,
    _reservation: Option<xc_core::MemoryReservation>,
}
fn prepared_residual_bytes(entries: usize, work: u32) -> u128 {
    // Two MPFR endpoints plus the exponent/Vec/owner bookkeeping. The point
    // allowance includes MPFR allocation headers and Rust Float/interval data.
    (entries as u128 * 2 + 16) * (u128::from(work).div_ceil(8) + 96)
}
impl<'a> BorderedResidualSource<'a> {
    pub(super) fn new(matrix: &'a [Float], eigenvalue: &'a Float, p: u32) -> Self {
        Self {
            matrix,
            eigenvalue,
            p,
            source_precision: None,
            prepared: None,
        }
    }
    pub(super) fn clear_prepared(&mut self) -> bool {
        self.prepared.take().is_some()
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn evaluate(
        &mut self,
        unit: &[Float],
        forcing: &[Float],
        response: &[Float],
        multiplier: &Float,
        supplied_norm: &Float,
    ) -> Result<Float> {
        self.evaluate_with_policy(unit, forcing, response, multiplier, supplied_norm, None)
    }
    #[allow(clippy::too_many_arguments)]
    fn evaluate_with_policy(
        &mut self,
        unit: &[Float],
        forcing: &[Float],
        response: &[Float],
        multiplier: &Float,
        supplied_norm: &Float,
        policy: Option<GuardPolicy>,
    ) -> Result<Float> {
        let n = unit.len();
        if n == 0
            || n > 16385
            || n.checked_mul(n) != Some(self.matrix.len())
            || forcing.len() != n
            || response.len() != n
        {
            bail!("bordered response residual requires compatible nonempty dimensions");
        }
        let source_base = match self.source_precision {
            Some(base) => base,
            None => {
                let base = source_precision(
                    self.matrix.iter().chain(std::iter::once(self.eigenvalue)),
                    self.p,
                )?;
                self.source_precision = Some(base);
                base
            }
        };
        let base = source_base.max(source_precision(
            unit.iter()
                .chain(forcing)
                .chain(response)
                .chain([multiplier, supplied_norm]),
            self.p,
        )?);
        point_workspace(base, 3 * n + 3)?;
        if supplied_norm < &0 {
            bail!("bordered response matrix norm is negative");
        }
        let policy = match policy {
            Some(policy) => policy,
            None => GuardPolicy::configured(self.p)?,
        };
        for guard in policy.guards() {
            let work = base + guard;
            dense_workspace(self.matrix.len(), 3 * n + 3, base, work, 6)?;
            if self.prepared.as_ref().is_some_and(|value| {
                value.work != work
                    || value.exponent_range != (rug::float::exp_min(), rug::float::exp_max())
                    || (value._reservation.is_none() && xc_core::RunResources::current().is_some())
            }) {
                self.prepared = None;
            }
            let scratch = dense_scratch_bytes(self.matrix.len(), 3 * n + 3, work, 6);
            // Preserve extended-attempt refusal, including its error text. A
            // retained optional optimization must not block the old admission.
            let mut _reservation = match reserve_extended_attempt(guard, scratch) {
                Err(_) if self.prepared.is_some() => {
                    self.prepared = None;
                    reserve_extended_attempt(guard, scratch)?
                }
                value => value?,
            };
            // The legacy prefix did not reserve its scratch. Using retained
            // data additionally requires full scratch admission, for every
            // call and run owner. Refusal evicts the slot and takes that prefix.
            // Full admission deliberately does not subtract another run's
            // retained reservation or assume owner identity from equal budgets.
            let can_retain = if guard <= 4096 {
                match reserve_point_workspace(scratch) {
                    Ok(value) => {
                        _reservation = value;
                        true
                    }
                    Err(_) => {
                        self.prepared = None;
                        false
                    }
                }
            } else {
                true
            };
            let points = |values: &[Float]| {
                values
                    .iter()
                    .map(|v| Scaled::point(v, work))
                    .collect::<Result<Vec<_>>>()
            };
            let unit = points(unit)?;
            let response = points(response)?;
            let forcing = points(forcing)?;
            let eigenvalue = Scaled::point(self.eigenvalue, work)?;
            let multiplier = Scaled::point(multiplier, work)?;
            let mut local = None;
            if self.prepared.is_none() {
                let retained = if can_retain {
                    let bytes = prepared_residual_bytes(self.matrix.len(), work);
                    if let Some(resources) = xc_core::RunResources::current() {
                        u64::try_from(bytes)
                            .ok()
                            .and_then(|bytes| resources.try_reserve_retained(bytes).ok().map(Some))
                    } else if bytes + scratch <= super::source_working_budget()? {
                        Some(None)
                    } else {
                        None
                    }
                } else {
                    None
                };
                // Optional scratch belongs only to an admitted retained route.
                // The uncached legacy prefix keeps its original admission.
                if retained.is_none() && guard <= 4096 {
                    _reservation = None;
                }
                let retain = retained.is_some();
                let prepared = prepare_bordered_residual(
                    self.matrix,
                    &eigenvalue,
                    n,
                    work,
                    retained.flatten(),
                )?;
                if retain {
                    self.prepared = Some(prepared);
                } else {
                    local = Some(prepared);
                }
            }
            let prepared = self
                .prepared
                .as_ref()
                .or(local.as_ref())
                .expect("prepared source");
            // A caller's cached point norm cannot inflate the acceptance denominator.
            let Some(replayed_norm) = rounded(&prepared.matrix_norm.materialize()?, self.p)? else {
                continue;
            };
            if replayed_norm != *supplied_norm {
                bail!("bordered response matrix norm does not match the exact stored source");
            }
            let mut residual = bordered_residual_rows(
                &prepared.shifted,
                &unit,
                &forcing,
                &response,
                &multiplier,
                work,
            )?;
            residual.push(Scaled::sum(
                &unit
                    .iter()
                    .zip(&response)
                    .map(|(x, y)| x.mul(y))
                    .collect::<Result<Vec<_>>>()?,
                work,
            )?);
            let numerator = Scaled::norm(&residual, work)?;
            let denominator = Scaled::sum(
                &[
                    prepared.matrix_norm.mul(&Scaled::norm(&response, work)?)?,
                    Scaled::norm(&forcing, work)?,
                    Scaled::new(
                        crate::ccm::retained_evidence::finite_math::abs(&multiplier.value)?,
                        multiplier.exponent,
                    )?,
                ],
                work,
            )?;
            let ratio = if denominator.zero() {
                numerator.materialize()?
            } else {
                if !denominator.value.is_strictly_positive() {
                    continue;
                }
                Scaled::new(
                    numerator.value.div(&denominator.value)?,
                    numerator.exponent - denominator.exponent,
                )?
                .materialize()?
            };
            let natural = ratio.upper().clone().max(&Float::with_val(work, 1));
            if !crate::ccm::retained_evidence::finite_math::narrow(
                std::slice::from_ref(&ratio),
                &natural,
                self.p,
            )? {
                continue;
            }
            let bound = Float::with_val_round(self.p, ratio.upper(), Round::Up).0;
            if !bound.is_finite() || bound < 0 || (bound.is_zero() && !ratio.upper().is_zero()) {
                bail!("bordered response residual bound exceeds the output range");
            }
            return Ok(bound);
        }
        bail!(
            "bordered response residual bound unresolved within {} guard bits",
            policy.maximum_guard_bits()
        )
    }
}
fn prepare_bordered_residual(
    matrix: &[Float],
    eigenvalue: &Scaled,
    n: usize,
    work: u32,
    reservation: Option<xc_core::MemoryReservation>,
) -> Result<PreparedBorderedResidual> {
    let shifted = matrix
        .iter()
        .enumerate()
        .map(|(index, v)| {
            let value = Scaled::point(v, work)?;
            if index / n == index % n {
                Scaled::sum(&[value, eigenvalue.neg()], work)
            } else {
                Ok(value)
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let matrix_norm = Scaled::norm(&shifted, work)?;
    Ok(PreparedBorderedResidual {
        work,
        exponent_range: (rug::float::exp_min(), rug::float::exp_max()),
        shifted,
        matrix_norm,
        _reservation: reservation,
    })
}
fn bordered_residual_rows(
    shifted: &[Scaled],
    unit: &[Scaled],
    forcing: &[Scaled],
    response: &[Scaled],
    multiplier: &Scaled,
    work: u32,
) -> Result<Vec<Scaled>> {
    let n = unit.len();
    let row = |(row, coefficients): (usize, &[Scaled])| {
        let mut terms = coefficients
            .iter()
            .zip(response)
            .map(|(x, y)| x.mul(y))
            .collect::<Result<Vec<_>>>()?;
        terms.push(unit[row].mul(multiplier)?);
        terms.push(Scaled::new(
            forcing[row].value.clone(),
            forcing[row].exponent,
        )?);
        Scaled::sum(&terms, work)
    };
    let resources = xc_core::RunResources::current();
    let parallel = n >= 16
        && resources
            .as_ref()
            .is_none_or(|owner| rayon::current_num_threads() <= owner.worker_limit())
        && xc_numerics::mpfr_interval::ensure_uniform_exponent_range().is_ok();
    if parallel {
        // Indexed collection retains row order and selects the first row error
        // in that order. Each row's products and directed sum remain unchanged.
        // At most n rows are live, so the existing six dense working floats per
        // entry cover shifted entries, row products and aligned sum scratch.
        let rows: Vec<Result<Scaled>> = shifted
            .par_chunks_exact(n)
            .enumerate()
            .map(|input| {
                let _scope = resources.as_ref().map(xc_core::RunResources::enter);
                row(input)
            })
            .collect();
        rows.into_iter().collect()
    } else {
        shifted.chunks_exact(n).enumerate().map(row).collect()
    }
}

/// An upper bound for the exact stored bordered residual divided by
/// ||A-lambda I||_F ||v||_2 + ||f||_2 + |mu|. The border row is u^T v.
/// This verifies that diagnostic, not continuum accuracy or forward solve error.
#[allow(clippy::too_many_arguments)]
pub(super) fn bordered_residual(
    matrix: &[Float],
    eigenvalue: &Float,
    unit: &[Float],
    forcing: &[Float],
    response: &[Float],
    multiplier: &Float,
    supplied_norm: &Float,
    p: u32,
) -> Result<Float> {
    BorderedResidualSource::new(matrix, eigenvalue, p).evaluate(
        unit,
        forcing,
        response,
        multiplier,
        supplied_norm,
    )
}

#[cfg(test)]
/// An upper bound for the exact stored bordered residual divided by
/// ||A-lambda I||_F ||v||_2 + ||f||_2 + |mu|. The border row is u^T v.
/// This verifies that diagnostic, not continuum accuracy or forward solve error.
#[allow(clippy::too_many_arguments)]
fn bordered_residual_reference(
    matrix: &[Float],
    eigenvalue: &Float,
    unit: &[Float],
    forcing: &[Float],
    response: &[Float],
    multiplier: &Float,
    supplied_norm: &Float,
    p: u32,
    policy: Option<GuardPolicy>,
) -> Result<Float> {
    let n = unit.len();
    if n == 0
        || n > 16385
        || n.checked_mul(n) != Some(matrix.len())
        || forcing.len() != n
        || response.len() != n
    {
        bail!("bordered response residual requires compatible nonempty dimensions");
    }
    let base = precision(
        matrix
            .iter()
            .chain(unit)
            .chain(forcing)
            .chain(response)
            .chain([eigenvalue, multiplier, supplied_norm]),
        p,
        3 * n + 3,
    )?;
    if supplied_norm < &0 {
        bail!("bordered response matrix norm is negative");
    }
    let policy = match policy {
        Some(policy) => policy,
        None => GuardPolicy::configured(p)?,
    };
    for guard in policy.guards() {
        let work = base + guard;
        dense_workspace(matrix.len(), 3 * n + 3, base, work, 6)?;
        let _reservation =
            reserve_extended_attempt(guard, dense_scratch_bytes(matrix.len(), 3 * n + 3, work, 6))?;
        let points = |values: &[Float]| {
            values
                .iter()
                .map(|v| Scaled::point(v, work))
                .collect::<Result<Vec<_>>>()
        };
        let unit = points(unit)?;
        let response = points(response)?;
        let forcing = points(forcing)?;
        let eigenvalue = Scaled::point(eigenvalue, work)?;
        let multiplier = Scaled::point(multiplier, work)?;
        let shifted = matrix
            .iter()
            .enumerate()
            .map(|(index, v)| {
                let value = Scaled::point(v, work)?;
                if index / n == index % n {
                    Scaled::sum(&[value, eigenvalue.neg()], work)
                } else {
                    Ok(value)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let matrix_norm = Scaled::norm(&shifted, work)?;
        // A caller's cached point norm cannot inflate the acceptance denominator.
        let Some(replayed_norm) = rounded(&matrix_norm.materialize()?, p)? else {
            continue;
        };
        if replayed_norm != *supplied_norm {
            bail!("bordered response matrix norm does not match the exact stored source");
        }
        let mut residual = Vec::with_capacity(n + 1);
        for (row, coefficients) in shifted.chunks_exact(n).enumerate() {
            let mut terms = coefficients
                .iter()
                .zip(&response)
                .map(|(x, y)| x.mul(y))
                .collect::<Result<Vec<_>>>()?;
            terms.push(unit[row].mul(&multiplier)?);
            terms.push(Scaled::new(
                forcing[row].value.clone(),
                forcing[row].exponent,
            )?);
            residual.push(Scaled::sum(&terms, work)?);
        }
        residual.push(Scaled::sum(
            &unit
                .iter()
                .zip(&response)
                .map(|(x, y)| x.mul(y))
                .collect::<Result<Vec<_>>>()?,
            work,
        )?);
        let numerator = Scaled::norm(&residual, work)?;
        let denominator = Scaled::sum(
            &[
                matrix_norm.mul(&Scaled::norm(&response, work)?)?,
                Scaled::norm(&forcing, work)?,
                Scaled::new(
                    crate::ccm::retained_evidence::finite_math::abs(&multiplier.value)?,
                    multiplier.exponent,
                )?,
            ],
            work,
        )?;
        let ratio = if denominator.zero() {
            numerator.materialize()?
        } else {
            if !denominator.value.is_strictly_positive() {
                continue;
            }
            Scaled::new(
                numerator.value.div(&denominator.value)?,
                numerator.exponent - denominator.exponent,
            )?
            .materialize()?
        };
        let natural = ratio.upper().clone().max(&Float::with_val(work, 1));
        if !crate::ccm::retained_evidence::finite_math::narrow(
            std::slice::from_ref(&ratio),
            &natural,
            p,
        )? {
            continue;
        }
        let bound = Float::with_val_round(p, ratio.upper(), Round::Up).0;
        if !bound.is_finite() || bound < 0 || (bound.is_zero() && !ratio.upper().is_zero()) {
            bail!("bordered response residual bound exceeds the output range");
        }
        return Ok(bound);
    }
    bail!(
        "bordered response residual bound unresolved within {} guard bits",
        policy.maximum_guard_bits()
    )
}

/// Correctly rounded a_i - u_i * eigenvalue_response for exact stored points.
pub(super) fn projected_forcing(
    action: &[Float],
    unit: &[Float],
    eigenvalue_response: &Float,
    p: u32,
) -> Result<Vec<Float>> {
    projected_forcing_with_policy(
        action,
        unit,
        eigenvalue_response,
        p,
        GuardPolicy::configured(p)?,
    )
}
fn projected_forcing_with_policy(
    action: &[Float],
    unit: &[Float],
    eigenvalue_response: &Float,
    p: u32,
    policy: GuardPolicy,
) -> Result<Vec<Float>> {
    if action.len() != unit.len() || action.len() > 16385 {
        bail!("response forcing projection shape is invalid");
    }
    let base = precision(
        action
            .iter()
            .chain(unit)
            .chain(std::iter::once(eigenvalue_response)),
        p,
        2 * action.len() + 1,
    )?;
    for guard in policy.guards() {
        let work = base + guard;
        let _reservation =
            reserve_extended_attempt(guard, vector_workspace_bytes(2 * action.len() + 1, work))?;
        let eigenvalue_response = Scaled::point(eigenvalue_response, work)?;
        let mut result = Vec::with_capacity(action.len());
        for (a, u) in action.iter().zip(unit) {
            let projection = Scaled::point(u, work)?.mul(&eigenvalue_response)?;
            let value =
                Scaled::sum(&[Scaled::point(a, work)?, projection.neg()], work)?.materialize()?;
            let Some(point) = rounded(&value, p)? else {
                break;
            };
            result.push(point);
        }
        if result.len() == action.len() {
            return Ok(result);
        }
    }
    bail!(
        "response forcing projection rounding unresolved within {} guard bits",
        policy.maximum_guard_bits()
    )
}

/// MPFR rounds the complete exact stored sum once, including cancellation.
pub(super) fn point_sum(values: &[Float], p: u32) -> Result<Float> {
    if values.len() > 16385 {
        bail!("response state sum shape exceeds the supported limit");
    }
    precision(values.iter(), p, values.len())?;
    let (value, direction) = Float::with_val_round(p, Float::sum(values.iter()), Round::Nearest);
    if !value.is_finite() || (value.is_zero() && direction != std::cmp::Ordering::Equal) {
        bail!("response state sum exceeds the output range");
    }
    Ok(value)
}

/// Correctly rounded (target_velocity - scale * sum(response)) / sum(unit).
/// All sums and scalar operands refer to exact stored points. The supplied
/// rounded unit sum is replayed; it is not substituted for the exact denominator.
pub(super) fn normalization_tangent(
    unit: &[Float],
    response: &[Float],
    target_velocity: &Float,
    normalization_scale: &Float,
    supplied_sum: &Float,
    p: u32,
) -> Result<Float> {
    if unit.is_empty() || unit.len() > 16385 || response.len() != unit.len() {
        bail!("response normalization tangent shape is invalid");
    }
    let base = precision(
        unit.iter()
            .chain(response)
            .chain([target_velocity, normalization_scale, supplied_sum]),
        p,
        2 * unit.len() + 3,
    )?;
    if *supplied_sum != point_sum(unit, p)? {
        bail!("response normalization sum does not match the stored state");
    }
    let policy = GuardPolicy::configured(p)?;
    for guard in policy.guards() {
        let work = base + guard;
        let _reservation =
            reserve_extended_attempt(guard, vector_workspace_bytes(2 * unit.len() + 3, work))?;
        let total = |v: &[Float]| {
            Scaled::sum(
                &v.iter()
                    .map(|x| Scaled::point(x, work))
                    .collect::<Result<Vec<_>>>()?,
                work,
            )
        };
        let denominator = total(unit)?;
        if denominator.zero() {
            bail!("response normalization has zero state sum");
        }
        if denominator.value.contains_zero() {
            continue;
        }
        let gauge = Scaled::point(normalization_scale, work)?.mul(&total(response)?)?;
        let numerator = Scaled::sum(&[Scaled::point(target_velocity, work)?, gauge.neg()], work)?;
        let result = Scaled::new(
            numerator.value.div(&denominator.value)?,
            numerator.exponent - denominator.exponent,
        )?
        .materialize()?;
        if let Some(point) = rounded(&result, p)? {
            return Ok(point);
        }
    }
    bail!(
        "response normalization tangent rounding unresolved within {} guard bits",
        policy.maximum_guard_bits()
    )
}

/// Correctly rounded components xi_i / sqrt(sum_j xi_j^2), for exact stored xi.
/// The returned rounded vector need not have an exactly unit rational norm.
pub(super) fn unit_state(xi: &[Float], p: u32) -> Result<Vec<Float>> {
    if xi.is_empty() || xi.len() > 16385 {
        bail!("response unit-state shape is invalid");
    }
    let base = precision(xi.iter(), p, xi.len())?;
    if xi.iter().all(Float::is_zero) {
        bail!("response unit state is zero");
    }
    let policy = GuardPolicy::configured(p)?;
    for guard in policy.guards() {
        let work = base + guard;
        let _reservation = reserve_extended_attempt(guard, vector_workspace_bytes(xi.len(), work))?;
        let points = xi
            .iter()
            .map(|v| Scaled::point(v, work))
            .collect::<Result<Vec<_>>>()?;
        let denominator = Scaled::norm(&points, work)?;
        if denominator.value.contains_zero() {
            continue;
        }
        let mut out = Vec::with_capacity(xi.len());
        for point in &points {
            let value = point.div(&denominator)?.materialize()?;
            if let Some(point) = rounded(&value, p)? {
                out.push(point);
            } else {
                break;
            }
        }
        if out.len() == xi.len() {
            return Ok(out);
        }
    }
    bail!(
        "response unit-state rounding unresolved within {} guard bits",
        policy.maximum_guard_bits()
    )
}

/// Correctly rounded sqrt(L) / exact_sum(unit). L and unit are stored points;
/// the separately rounded norm/sum must not be substituted into this formula.
pub(super) fn normalization_scale(unit: &[Float], l: &Float, p: u32) -> Result<Float> {
    if unit.is_empty() || unit.len() > 16385 {
        bail!("response normalization scale shape is invalid");
    }
    let base = precision(unit.iter().chain(std::iter::once(l)), p, unit.len() + 1)?;
    if l <= &0 {
        bail!("response normalization cutoff must be positive");
    }
    let policy = GuardPolicy::configured(p)?;
    for guard in policy.guards() {
        let work = base + guard;
        let _reservation =
            reserve_extended_attempt(guard, vector_workspace_bytes(unit.len() + 1, work))?;
        let denominator = Scaled::sum(
            &unit
                .iter()
                .map(|v| Scaled::point(v, work))
                .collect::<Result<Vec<_>>>()?,
            work,
        )?;
        if denominator.zero() {
            bail!("response normalization has zero state sum");
        }
        if denominator.value.contains_zero() {
            continue;
        }
        let value = Scaled::point(l, work)?
            .sqrt()?
            .div(&denominator)?
            .materialize()?;
        if let Some(value) = rounded(&value, p)? {
            return Ok(value);
        }
    }
    bail!(
        "response normalization scale rounding unresolved within {} guard bits",
        policy.maximum_guard_bits()
    )
}

/// Correctly rounded 1 / (2 sqrt(L)) for the exact stored positive cutoff L.
pub(super) fn target_velocity(l: &Float, p: u32) -> Result<Float> {
    let base = precision(std::iter::once(l), p, 1)?;
    if l <= &0 {
        bail!("response normalization cutoff must be positive");
    }
    let policy = GuardPolicy::configured(p)?;
    for guard in policy.guards() {
        let work = base + guard;
        let _reservation = reserve_extended_attempt(guard, vector_workspace_bytes(1, work))?;
        let mut denominator = Scaled::point(l, work)?.sqrt()?;
        denominator.exponent += 1;
        let value = Scaled::point(&Float::with_val(work, 1), work)?
            .div(&denominator)?
            .materialize()?;
        if let Some(value) = rounded(&value, p)? {
            return Ok(value);
        }
    }
    bail!(
        "response target velocity rounding unresolved within {} guard bits",
        policy.maximum_guard_bits()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use rug::{float::Special, Rational};

    fn severe_dot_cancellation() -> ([Float; 2], [Float; 2]) {
        let p = 8192;
        let epsilon = Float::with_val(p, 1) >> (p - 1);
        (
            [Float::with_val(p, 1 + &epsilon), Float::with_val(p, 1)],
            [Float::with_val(p, 1 - &epsilon), Float::with_val(p, -1)],
        )
    }

    #[test]
    fn response_dot_exact_fallback_resolves_cancellation_beyond_legacy_guards() {
        let (left, right) = severe_dot_cancellation();
        let resources = xc_core::RunResources::new(1 << 20).unwrap();
        let _scope = resources.enter();
        assert!(interval_dot(&left, &right, 8192, 8192, 1, 1)
            .unwrap()
            .is_none());
        let exact = left
            .iter()
            .zip(&right)
            .fold(Rational::from(0), |acc, (a, b)| {
                acc + a.to_rational().unwrap() * b.to_rational().unwrap()
            });
        let result = dot(&left, &right, 8192).unwrap();
        assert_eq!(result, Float::with_val(8192, exact));
        assert!(!result.is_zero());
        assert!(resources.usage().peak_bytes > 0);
        assert_eq!(resources.usage().working_bytes, 0);
    }

    #[test]
    fn response_dot_exact_fallback_ties_zero_and_legacy_bytes() {
        for p in [64, 128, 256] {
            for numerator in [-3, -1, 0, 1, 3] {
                let left = [Float::with_val(p, 1), Float::with_val(p, numerator) >> p];
                let right = [Float::with_val(p, 1), Float::with_val(p, 1)];
                let exact = left
                    .iter()
                    .zip(&right)
                    .fold(Rational::from(0), |acc, (a, b)| {
                        acc + a.to_rational().unwrap() * b.to_rational().unwrap()
                    });
                let result = exact_dot(&left, &right, p, 1, 1).unwrap();
                assert_eq!(result, Float::with_val(p, exact));
                let old = interval_dot(&left, &right, p, p, 1, 1).unwrap().unwrap();
                assert_eq!(
                    super::super::lossless_hp_decimal(&old),
                    super::super::lossless_hp_decimal(&dot(&left, &right, p).unwrap())
                );
            }
            let left = [Float::with_val(p, 1), Float::with_val(p, -1)];
            let right = [Float::with_val(p, 1), Float::with_val(p, 1)];
            let zero = exact_dot(&left, &right, p, 1, 1).unwrap();
            assert!(zero.is_zero());
            assert!(!zero.is_sign_negative());
        }
    }

    #[test]
    fn response_dot_exact_midpoints_and_zero_beyond_legacy_guards() {
        let source_p = 8192;
        let epsilon = Float::with_val(source_p, 1) >> (source_p - 1);
        for numerator in [1, 3] {
            for sign in [-1, 1] {
                // The first two exact products sum to one, but their narrow
                // interval straddles it until the 16382-bit term is retained.
                let left = [
                    Float::with_val(source_p, 1 + &epsilon) * sign,
                    Float::with_val(source_p, &epsilon * sign),
                    Float::with_val(source_p, numerator * sign) >> 64u32,
                ];
                let right = [
                    Float::with_val(source_p, 1 - &epsilon),
                    epsilon.clone(),
                    Float::with_val(source_p, 1),
                ];
                assert!(interval_dot(&left, &right, 64, source_p, 1, 1)
                    .unwrap()
                    .is_none());
                let exact = left
                    .iter()
                    .zip(&right)
                    .fold(Rational::from(0), |acc, (a, b)| {
                        acc + a.to_rational().unwrap() * b.to_rational().unwrap()
                    });
                assert_eq!(dot(&left, &right, 64).unwrap(), Float::with_val(64, exact));
            }
        }
        let left = [
            Float::with_val(source_p, 1 + &epsilon),
            epsilon.clone(),
            Float::with_val(source_p, 1),
        ];
        let right = [
            Float::with_val(source_p, 1 - &epsilon),
            epsilon,
            Float::with_val(source_p, -1),
        ];
        assert!(interval_dot(&left, &right, 64, source_p, 1, 1)
            .unwrap()
            .is_none());
        let zero = dot(&left, &right, 64).unwrap();
        assert!(zero.is_zero());
        assert!(!zero.is_sign_negative());
    }

    #[test]
    fn response_dot_exact_fallback_admission_and_exponent_refusals_release_ownership() {
        let (left, right) = severe_dot_cancellation();
        let resources = xc_core::RunResources::new(1 << 20).unwrap();
        let _scope = resources.enter();
        let occupied = resources.try_reserve_retained((1 << 20) - 1).unwrap();
        assert!(dot(&left, &right, 8192)
            .unwrap_err()
            .to_string()
            .contains("memory budget"));
        assert_eq!(resources.usage().working_bytes, 0);
        assert_eq!(resources.usage().retained_bytes, (1 << 20) - 1);
        drop(occupied);
        for exponent in [-700_000_000i64, 700_000_000] {
            let value = scale_float(&Float::with_val(128, 1), exponent, 128).unwrap();
            assert!(exact_dot(
                std::slice::from_ref(&value),
                std::slice::from_ref(&value),
                128,
                exponent + 1,
                exponent + 1
            )
            .is_err());
            assert_eq!(resources.usage().working_bytes, 0);
        }
        // An extreme component span is refused before constructing products.
        let wide = [
            Float::with_val(128, 1),
            Float::with_val(128, 1) >> 700_000_000u32,
        ];
        assert!(exact_dot(&wide, &wide, 128, 1, 1)
            .unwrap_err()
            .to_string()
            .contains("memory budget"));
        assert_eq!(resources.usage().working_bytes, 0);
    }

    #[test]
    fn response_projection_extended_guards_are_admitted_and_preserve_exact_rounding() {
        let (left, right) = severe_dot_cancellation();
        let action = [Float::with_val(8192, 1)];
        let unit = &left[..1];
        let eigenvalue = &right[0];
        let old = GuardPolicy::new(8192, 4096).unwrap();
        assert!(
            projected_forcing_with_policy(&action, unit, eigenvalue, 8192, old)
                .unwrap_err()
                .to_string()
                .contains("4096")
        );
        let policy = GuardPolicy::new(8192, 8192).unwrap();
        let resources = xc_core::RunResources::new(1 << 20).unwrap();
        let _scope = resources.enter();
        let result =
            projected_forcing_with_policy(&action, unit, eigenvalue, 8192, policy).unwrap();
        let exact =
            Rational::from(1) - unit[0].to_rational().unwrap() * eigenvalue.to_rational().unwrap();
        assert_eq!(result, vec![Float::with_val(8192, exact)]);
        assert!(resources.usage().peak_bytes > 0);
        assert_eq!(resources.usage().working_bytes, 0);
        let _occupied = resources.try_reserve_retained((1 << 20) - 1).unwrap();
        assert!(
            projected_forcing_with_policy(&action, unit, eigenvalue, 8192, policy)
                .unwrap_err()
                .to_string()
                .contains("memory budget")
        );
        assert_eq!(resources.usage().working_bytes, 0);
    }

    #[test]
    fn response_normalization_exact_rational_oracles() {
        for p in [64, 128, 256] {
            for i in -9..=9 {
                for j in -5..=5 {
                    let f = |n, d| Float::with_val(320, Rational::from((n, d)));
                    let unit = [f(i, 7), f(1, 3)];
                    let response = [f(j, 11), f(i - j, 13)];
                    let target = f(i + 1, 17);
                    let scale = f(j + 2, 19);
                    let sum = unit
                        .iter()
                        .fold(Rational::from(0), |a, x| a + x.to_rational().unwrap());
                    assert_ne!(sum, 0);
                    let response_sum = response
                        .iter()
                        .fold(Rational::from(0), |a, x| a + x.to_rational().unwrap());
                    let expected = Float::with_val(
                        p,
                        (target.to_rational().unwrap()
                            - scale.to_rational().unwrap() * response_sum)
                            / &sum,
                    );
                    let stored = point_sum(&unit, p).unwrap();
                    assert_eq!(stored, Float::with_val(p, sum));
                    assert_eq!(
                        normalization_tangent(&unit, &response, &target, &scale, &stored, p)
                            .unwrap(),
                        expected,
                        "p={p},i={i},j={j}"
                    );
                }
            }
        }
    }
    #[test]
    fn response_sum_cancels_unrepresentable_partial_sum_and_preserves_source_bits() {
        let p = 128;
        let huge = Float::with_val(p, 0.75) << rug::float::exp_max();
        assert_eq!(
            point_sum(&[huge.clone(), huge.clone(), -huge.clone()], p).unwrap(),
            huge
        );
        let one = Float::with_val(320, 1);
        let delta = one.clone() >> 250u32;
        assert_eq!(
            point_sum(&[Float::with_val(320, &one + &delta), -one], 64).unwrap(),
            delta
        );
        assert_eq!(point_sum(&[], 64).unwrap(), 0);
    }
    #[test]
    fn response_normalization_validates_all_sources_and_shape() {
        let p = 128;
        let one = Float::with_val(p, 1);
        for slot in 0..5 {
            let mut values = vec![one.clone(); 5];
            values[slot] = Float::with_val(p, Special::Nan);
            assert!(normalization_tangent(
                &values[0..1],
                &values[1..2],
                &values[2],
                &values[3],
                &values[4],
                p
            )
            .is_err());
        }
        for p in [0, 63, 1_000_001, u32::MAX] {
            assert!(point_sum(&[], p).is_err());
            assert!(normalization_tangent(
                std::slice::from_ref(&one),
                std::slice::from_ref(&one),
                &one,
                &one,
                &one,
                p
            )
            .is_err());
        }
        assert!(
            normalization_tangent(std::slice::from_ref(&one), &[], &one, &one, &one, p).is_err()
        );
        assert!(normalization_tangent(&[], &[], &one, &one, &one, p).is_err());
        assert!(point_sum(&vec![one.clone(); 16386], p).is_err());
        assert!(point_sum(&[Float::with_val(1_000_129, 1)], p).is_err());
        let huge = Float::with_val(p, 0.75) << rug::float::exp_max();
        assert!(point_sum(&[huge.clone(), huge], p).is_err());
        let minimum = Float::with_val(p, 1) << (rug::float::exp_min() - 1);
        let mut adjacent = minimum.clone();
        adjacent.next_up();
        assert!(point_sum(&[adjacent, -minimum], p).is_err());
    }
    #[test]
    fn response_projection_exact_rational_oracles() {
        for p in [64, 128, 256] {
            for i in -9..=9 {
                for j in -5..=5 {
                    let f = |n, d| Float::with_val(320, Rational::from((n, d)));
                    let action = [f(i, 7), f(j, 17)];
                    let unit = [f(j, 11), f(i, 13)];
                    let eigenvalue = f(i - j, 19);
                    let expected = action
                        .iter()
                        .zip(&unit)
                        .map(|(a, u)| {
                            Float::with_val(
                                p,
                                a.to_rational().unwrap()
                                    - u.to_rational().unwrap() * eigenvalue.to_rational().unwrap(),
                            )
                        })
                        .collect::<Vec<_>>();
                    assert_eq!(
                        projected_forcing(&action, &unit, &eigenvalue, p).unwrap(),
                        expected
                    );
                }
            }
        }
    }
    #[test]
    fn response_projection_invalid_precision_range_and_empty_domain() {
        let one = Float::with_val(128, 1);
        for p in [0, 63, 1_000_001, u32::MAX] {
            assert!(projected_forcing(&[], &[], &one, p).is_err());
        }
        assert!(projected_forcing(&[], &[], &one, 64).unwrap().is_empty());
        for special in [Special::Nan, Special::Infinity, Special::NegInfinity] {
            let bad = Float::with_val(128, special);
            assert!(projected_forcing(
                std::slice::from_ref(&bad),
                std::slice::from_ref(&one),
                &one,
                128
            )
            .is_err());
            assert!(projected_forcing(
                std::slice::from_ref(&one),
                std::slice::from_ref(&one),
                &bad,
                128
            )
            .is_err());
        }
        let huge = one.clone() << 700_000_000u32;
        let tiny = one.clone() >> 700_000_000u32;
        assert!(projected_forcing(
            &[Float::with_val(128, 0)],
            std::slice::from_ref(&huge),
            &huge,
            128
        )
        .is_err());
        assert!(projected_forcing(
            &[Float::with_val(128, 0)],
            std::slice::from_ref(&tiny),
            &tiny,
            128
        )
        .is_err());
    }
    #[test]
    fn bordered_residual_rational_oracles() {
        fn square_sum(v: &[Rational]) -> Rational {
            v.iter()
                .fold(Rational::from(0), |acc, x| acc + Rational::from(x * x))
        }
        fn norm(v: &[Rational]) -> I {
            I::from_rational(&square_sum(v), 4096).sqrt().unwrap()
        }
        for p in [64, 128, 256] {
            for i in -3..=3 {
                for j in -3..=3 {
                    for exponent in [-100i64, 0, 100] {
                        let f = |n, d| Float::with_val(320, Rational::from((n, d)));
                        let scale = |x: Float| scale_float(&x, exponent, 320).unwrap();
                        let matrix = [
                            scale(f(i, 3)),
                            scale(f(j, 7)),
                            scale(f(j, 7)),
                            scale(f(i + 1, 11)),
                        ];
                        let eigenvalue = scale(f(1, 5));
                        let unit = [f(3, 5), f(4, 5)];
                        let response = [f(i + 2, 13), f(j - 1, 17)];
                        let forcing = [scale(f(j + 2, 19)), scale(f(i - 3, 23))];
                        let multiplier = scale(f(i - j, 29));
                        let source_norm = shifted_norm(&matrix, &eigenvalue, 2, p).unwrap();
                        let actual = bordered_residual(
                            &matrix,
                            &eigenvalue,
                            &unit,
                            &forcing,
                            &response,
                            &multiplier,
                            &source_norm,
                            p,
                        )
                        .unwrap();
                        let rationals = |v: &[Float]| {
                            v.iter()
                                .map(|x| x.to_rational().unwrap())
                                .collect::<Vec<_>>()
                        };
                        let mut a = rationals(&matrix);
                        let eigenvalue = eigenvalue.to_rational().unwrap();
                        a[0] -= &eigenvalue;
                        a[3] -= &eigenvalue;
                        let u = rationals(&unit);
                        let v = rationals(&response);
                        let f = rationals(&forcing);
                        let m = multiplier.to_rational().unwrap();
                        let residual = [
                            Rational::from(&a[0] * &v[0])
                                + Rational::from(&a[1] * &v[1])
                                + Rational::from(&u[0] * &m)
                                + &f[0],
                            Rational::from(&a[2] * &v[0])
                                + Rational::from(&a[3] * &v[1])
                                + Rational::from(&u[1] * &m)
                                + &f[1],
                            Rational::from(&u[0] * &v[0]) + Rational::from(&u[1] * &v[1]),
                        ];
                        let denominator = norm(&a)
                            .mul(&norm(&v))
                            .add(&norm(&f))
                            .add(&I::from_rational(&m.abs(), 4096));
                        let oracle = norm(&residual).div(&denominator).unwrap();
                        let expected = Float::with_val_round(p, oracle.upper(), Round::Up).0;
                        assert_eq!(
                            expected,
                            Float::with_val_round(p, oracle.lower(), Round::Up).0,
                            "oracle must resolve target rounding"
                        );
                        let mut slack = expected.clone();
                        slack.next_up();
                        assert!(
                            actual >= expected && actual <= slack,
                            "p={p},i={i},j={j},scale={exponent}: bound {actual}, exact upper {expected}"
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn bordered_residual_cancellation_beyond_product_range() {
        let p = 128;
        let zero = Float::with_val(p, 0);
        let huge = Float::with_val(p, 1) << 700_000_000u32;
        let matrix = [huge.clone(), huge.clone(), huge.clone(), huge.clone()];
        let unit = [Float::with_val(p, 1), Float::with_val(p, 1)];
        let response = [huge.clone(), -huge.clone()];
        let supplied = Float::with_val(p, &huge * 2);
        assert_eq!(
            bordered_residual(
                &matrix,
                &zero,
                &unit,
                &[zero.clone(), zero.clone()],
                &response,
                &zero,
                &supplied,
                p
            )
            .unwrap(),
            0
        );
    }
    #[test]
    fn bordered_residual_zero_denominator_and_exact_solve() {
        let p = 128;
        let zero = Float::with_val(p, 0);
        let one = Float::with_val(p, 1);
        assert_eq!(
            bordered_residual(
                std::slice::from_ref(&zero),
                &zero,
                std::slice::from_ref(&one),
                std::slice::from_ref(&zero),
                std::slice::from_ref(&one),
                &zero,
                &zero,
                p
            )
            .unwrap(),
            1
        );
        assert_eq!(
            bordered_residual(
                std::slice::from_ref(&zero),
                &zero,
                &[one],
                &[Float::with_val(p, -1)],
                std::slice::from_ref(&zero),
                &Float::with_val(p, 1),
                &zero,
                p
            )
            .unwrap(),
            0
        );
    }
    #[test]
    fn bordered_residual_validates_every_scalar_and_dimension() {
        let p = 128;
        let zero = Float::with_val(p, 0);
        let one = Float::with_val(p, 1);
        for slot in 0..7 {
            let mut data = vec![one.clone(); 7];
            data[slot] = Float::with_val(p, Special::Nan);
            assert!(bordered_residual(
                &data[0..1],
                &data[1],
                &data[2..3],
                &data[3..4],
                &data[4..5],
                &data[5],
                &data[6],
                p
            )
            .is_err());
        }
        assert!(bordered_residual(
            std::slice::from_ref(&one),
            &zero,
            std::slice::from_ref(&one),
            &[],
            std::slice::from_ref(&one),
            &zero,
            &one,
            p
        )
        .is_err());
        assert!(bordered_residual(
            std::slice::from_ref(&one),
            &zero,
            std::slice::from_ref(&one),
            std::slice::from_ref(&zero),
            &[],
            &zero,
            &one,
            p
        )
        .is_err());
        for p in [0, 63, 1_000_001, u32::MAX] {
            assert!(bordered_residual(
                std::slice::from_ref(&one),
                &zero,
                std::slice::from_ref(&one),
                std::slice::from_ref(&zero),
                std::slice::from_ref(&one),
                &zero,
                &one,
                p
            )
            .is_err());
        }
        assert!(bordered_residual(
            std::slice::from_ref(&one),
            &zero,
            std::slice::from_ref(&one),
            std::slice::from_ref(&zero),
            std::slice::from_ref(&one),
            &zero,
            &Float::with_val(p, -1),
            p
        )
        .is_err());
    }

    #[test]
    fn response_point_exact_dot_rational_oracles() {
        for p in [64, 128, 256] {
            for i in -9..=9 {
                for j in -5..=5 {
                    let a = [
                        Float::with_val(320, Rational::from((i, 7))),
                        Float::with_val(320, Rational::from((j, 11))),
                        Float::with_val(320, 1),
                    ];
                    let b = [
                        Float::with_val(320, Rational::from((j, 3))),
                        Float::with_val(320, Rational::from((i, 13))),
                        Float::with_val(320, -1),
                    ];
                    let exact = a.iter().zip(&b).fold(Rational::from(0), |acc, (x, y)| {
                        acc + x.to_rational().unwrap() * y.to_rational().unwrap()
                    });
                    assert_eq!(
                        dot(&a, &b, p).unwrap(),
                        Float::with_val(p, exact),
                        "p={p},i={i},j={j}"
                    );
                }
            }
        }
    }
    #[test]
    fn response_point_scaled_pythagorean_norms() {
        for p in [64, 128, 256] {
            for exponent in [-700_000_000i64, -100, 0, 100, 700_000_000] {
                let unit = scale_float(&Float::with_val(p, 1), exponent, p).unwrap();
                let values = [
                    Float::with_val(p, &unit * -3),
                    Float::with_val(p, &unit * 4),
                ];
                assert_eq!(norm(&values, p).unwrap(), Float::with_val(p, &unit * 5));
                assert_eq!(
                    dot(&values, &[Float::with_val(p, 4), Float::with_val(p, 3)], p).unwrap(),
                    0
                );
            }
        }
    }
    #[test]
    fn response_point_empty_vectors_and_zero_matrix() {
        assert_eq!(norm(&[], 64).unwrap(), 0);
        assert_eq!(dot(&[], &[], 64).unwrap(), 0);
        assert_eq!(
            shifted_norm(
                &vec![Float::with_val(128, 0); 9],
                &Float::with_val(128, 0),
                3,
                64
            )
            .unwrap(),
            0
        );
    }
    #[test]
    fn response_point_domain_errors_are_explicit() {
        let zero = Float::with_val(64, 0);
        let one = Float::with_val(64, 1);
        for p in [0, 1, 63, 1_000_001, u32::MAX] {
            assert!(norm(&[], p).is_err());
            assert!(dot(&[], &[], p).is_err());
            assert!(shifted_norm(std::slice::from_ref(&one), &zero, 1, p).is_err());
        }
        for special in [Special::Nan, Special::Infinity, Special::NegInfinity] {
            let value = Float::with_val(64, special);
            assert!(norm(std::slice::from_ref(&value), 64).is_err());
            assert!(dot(std::slice::from_ref(&value), std::slice::from_ref(&one), 64).is_err());
            assert!(shifted_norm(std::slice::from_ref(&one), &value, 1, 64).is_err());
        }
        for dimension in [0, 2, 16386, usize::MAX] {
            assert!(shifted_norm(std::slice::from_ref(&one), &zero, dimension, 64).is_err());
        }
        assert!(norm(&vec![zero.clone(); 16386], 64).is_err());
        let overprec = Float::with_val(1_000_129, 1);
        assert!(norm(std::slice::from_ref(&overprec), 64).is_err());
        assert!(precision(std::iter::empty(), 1_000_000, usize::MAX).is_err());
    }
    #[test]
    fn response_point_range_failures_cannot_be_finite_zero() {
        let huge = Float::with_val(128, 1) << 700_000_000u32;
        let tiny = Float::with_val(128, 1) >> 700_000_000u32;
        assert!(dot(
            std::slice::from_ref(&huge),
            std::slice::from_ref(&huge),
            128
        )
        .is_err());
        assert!(dot(
            std::slice::from_ref(&tiny),
            std::slice::from_ref(&tiny),
            128
        )
        .is_err());
        assert!(norm(&[huge, tiny], 128).is_err()); // exact rescaling cannot retain both components
    }
}

#[cfg(test)]
mod resource_tests {
    use super::*;
    #[test]
    fn documented_response_shape_uses_actual_dense_buffer_lifetimes() {
        for (n, p) in [(661usize, 131u32), (801, 6708), (1001, 3386)] {
            for guard in GUARDS {
                assert!(dense_workspace_bytes(n * n, 3 * n + 3, p, p + guard, 6) < (8u128 << 30));
            }
        }
        assert!(dense_workspace(16385 * 16385, 3 * 16385 + 3, 1_000_000, 1_004_096, 6).is_err());
        for p in [131u32, 6708] {
            let matrix = [2, 0, 0, 0, 3, 0, 0, 0, 6].map(|x| Float::with_val(p, x));
            let zero = Float::with_val(p, 0);
            assert_eq!(shifted_norm(&matrix, &zero, 3, p).unwrap(), 7);
            let unit = [0, 1, 0].map(|x| Float::with_val(p, x));
            let vector = [0, 0, 0].map(|x| Float::with_val(p, x));
            assert_eq!(
                bordered_residual(
                    &matrix,
                    &zero,
                    &unit,
                    &vector,
                    &vector,
                    &zero,
                    &Float::with_val(p, 7),
                    p
                )
                .unwrap(),
                0
            );
        }
    }
}

#[cfg(test)]
mod prepared_bordered_tests {
    use super::*;
    use rug::{float::Special, Rational};

    fn same_point(actual: &Float, expected: &Float) {
        assert_eq!(actual, expected);
        assert_eq!(actual.prec(), expected.prec());
        assert_eq!(actual.is_sign_negative(), expected.is_sign_negative());
        assert_eq!(actual.to_integer_exp(), expected.to_integer_exp());
    }
    fn fixture(n: usize, p: u32) -> (Vec<Float>, Float, Vec<Float>, Vec<Float>, Vec<Float>) {
        let f = |a: i32, b: i32| Float::with_val(p, Rational::from((a, b)));
        let matrix = (0..n * n)
            .map(|index| {
                let (row, col) = (index / n, index % n);
                if row == col {
                    f(row as i32 + 2, 3)
                } else if (row + col) % 5 == 0 {
                    f((row as i32 - col as i32) % 7, 17)
                } else {
                    Float::with_val(
                        p,
                        if index % 2 == 0 {
                            Special::Zero
                        } else {
                            Special::NegZero
                        },
                    )
                }
            })
            .collect();
        (
            matrix,
            f(1, 7),
            (0..n).map(|i| f(i as i32 % 3 - 1, 5)).collect(),
            (0..n).map(|i| f(i as i32 % 5 - 2, 13)).collect(),
            (0..n).map(|i| f(i as i32 % 7 - 3, 19)).collect(),
        )
    }

    #[test]
    fn prepared_bordered_reuses_source_and_preserves_old_point_bits_across_threads() {
        for threads in [1, 2, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let resources =
                    xc_core::RunResources::with_worker_limit(64 << 20, threads).unwrap();
                let _scope = resources.enter();
                for p in [64, 128, 256] {
                    let (matrix, eigenvalue, unit, mut forcing, response) = fixture(17, p + 64);
                    let multiplier = Float::with_val(p, Rational::from((2, 23)));
                    let supplied = shifted_norm(&matrix, &eigenvalue, 17, p).unwrap();
                    let mut source = BorderedResidualSource::new(&matrix, &eigenvalue, p);
                    let mut address = None;
                    for event in 0..4 {
                        forcing[event] = -forcing[event].clone();
                        let expected = bordered_residual_reference(
                            &matrix,
                            &eigenvalue,
                            &unit,
                            &forcing,
                            &response,
                            &multiplier,
                            &supplied,
                            p,
                            None,
                        )
                        .unwrap();
                        let actual = source
                            .evaluate(&unit, &forcing, &response, &multiplier, &supplied)
                            .unwrap();
                        same_point(&actual, &expected);
                        let prepared = source.prepared.as_ref().unwrap();
                        assert_eq!(prepared.work, p + 128);
                        let current = prepared.shifted.as_ptr();
                        if let Some(previous) = address {
                            assert_eq!(current, previous);
                        }
                        address = Some(current);
                        assert_eq!(resources.usage().working_bytes, 0);
                        assert!(resources.usage().retained_bytes > 0);
                    }
                    let mut wrong = supplied.clone();
                    wrong.next_up();
                    let expected = bordered_residual_reference(
                        &matrix,
                        &eigenvalue,
                        &unit,
                        &forcing,
                        &response,
                        &multiplier,
                        &wrong,
                        p,
                        None,
                    )
                    .unwrap_err()
                    .to_string();
                    assert_eq!(
                        source
                            .evaluate(&unit, &forcing, &response, &multiplier, &wrong)
                            .unwrap_err()
                            .to_string(),
                        expected
                    );
                    drop(source);
                    assert_eq!(resources.usage().retained_bytes, 0);
                }
            });
        }
    }

    #[test]
    fn prepared_bordered_preserves_interval_endpoints_and_signed_zero() {
        for p in [64, 128, 256] {
            let (matrix, eigenvalue, _, _, _) = fixture(3, p + 64);
            let work = p + 128;
            let eigenvalue = Scaled::point(&eigenvalue, work).unwrap();
            let prepared = prepare_bordered_residual(&matrix, &eigenvalue, 3, work, None).unwrap();
            let reference = matrix
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    let value = Scaled::point(value, work)?;
                    if index / 3 == index % 3 {
                        Scaled::sum(&[value, eigenvalue.neg()], work)
                    } else {
                        Ok(value)
                    }
                })
                .collect::<Result<Vec<_>>>()
                .unwrap();
            let reference_norm = Scaled::norm(&reference, work).unwrap();
            for (actual, expected) in prepared
                .shifted
                .iter()
                .zip(&reference)
                .chain(std::iter::once((&prepared.matrix_norm, &reference_norm)))
            {
                assert_eq!(actual.exponent, expected.exponent);
                same_point(actual.value.lower(), expected.value.lower());
                same_point(actual.value.upper(), expected.value.upper());
            }
            let zero = Float::with_val(p, Special::NegZero);
            let one = Float::with_val(p, 1);
            let expected = bordered_residual_reference(
                std::slice::from_ref(&zero),
                &zero,
                std::slice::from_ref(&one),
                std::slice::from_ref(&zero),
                std::slice::from_ref(&zero),
                &zero,
                &zero,
                p,
                None,
            )
            .unwrap();
            let actual = bordered_residual(
                std::slice::from_ref(&zero),
                &zero,
                std::slice::from_ref(&one),
                std::slice::from_ref(&zero),
                std::slice::from_ref(&zero),
                &zero,
                &zero,
                p,
            )
            .unwrap();
            same_point(&actual, &expected);
        }
    }

    #[test]
    fn prepared_bordered_preserves_extreme_exponent_cancellation_and_source_bits() {
        for (p, source_p, exponent) in [
            (128, 128, 700_000_000i64),
            (128, 128, -700_000_000),
            (64, 320, 0),
        ] {
            let zero = Float::with_val(source_p, 0);
            let one = Float::with_val(source_p, 1);
            let magnitude = scale_float(&one, exponent, source_p).unwrap();
            let matrix = vec![magnitude.clone(); 4];
            let unit = [one.clone(), one];
            let response = [magnitude.clone(), -magnitude.clone()];
            let vector = [zero.clone(), zero.clone()];
            let supplied = Float::with_val(p, &magnitude * 2);
            let expected = bordered_residual_reference(
                &matrix, &zero, &unit, &vector, &response, &zero, &supplied, p, None,
            )
            .unwrap();
            let mut source = BorderedResidualSource::new(&matrix, &zero, p);
            for _ in 0..2 {
                same_point(
                    &source
                        .evaluate(&unit, &vector, &response, &zero, &supplied)
                        .unwrap(),
                    &expected,
                );
            }
        }
    }

    // sqrt((1-2^-p)^2 + (2^(1-p/2))^2) = 1+2^-p is
    // exactly a p-bit midpoint. The low square bit forces guard escalation.
    fn midpoint_matrix(p: u32) -> [Float; 4] {
        let one = Float::with_val(p, 1);
        let epsilon = one.clone() >> p;
        [
            Float::with_val(p, &one - epsilon),
            one >> (p / 2 - 1),
            Float::with_val(p, 0),
            Float::with_val(p, 0),
        ]
    }

    #[test]
    fn prepared_bordered_keys_actual_work_and_preserves_guard_escalation() {
        let p = 64;
        let resources = xc_core::RunResources::new(32 << 20).unwrap();
        let _scope = resources.enter();
        let matrix = midpoint_matrix(p);
        let zero = Float::with_val(p, 0);
        let one = Float::with_val(p, 1);
        let unit = [one.clone(), zero.clone()];
        let vector = [zero.clone(), zero.clone()];
        let first = prepare_bordered_residual(
            &matrix,
            &Scaled::point(&zero, p + 64).unwrap(),
            2,
            p + 64,
            None,
        )
        .unwrap();
        assert!(rounded(&first.matrix_norm.materialize().unwrap(), p)
            .unwrap()
            .is_none());
        let mut source = BorderedResidualSource::new(&matrix, &zero, p);
        let expected = bordered_residual_reference(
            &matrix, &zero, &unit, &vector, &vector, &zero, &one, p, None,
        )
        .unwrap();
        same_point(
            &source
                .evaluate(&unit, &vector, &vector, &zero, &one)
                .unwrap(),
            &expected,
        );
        assert_eq!(source.prepared.as_ref().unwrap().work, p + 128);
        // Output p is unchanged, but one higher-precision zero changes actual work.
        let higher = [Float::with_val(320, 0), zero.clone()];
        let expected = bordered_residual_reference(
            &matrix, &zero, &unit, &vector, &higher, &zero, &one, p, None,
        )
        .unwrap();
        same_point(
            &source
                .evaluate(&unit, &vector, &higher, &zero, &one)
                .unwrap(),
            &expected,
        );
        assert_eq!(source.prepared.as_ref().unwrap().work, 384);
        // A return to low-precision inputs also replaces the higher entry.
        source
            .evaluate(&unit, &vector, &vector, &zero, &one)
            .unwrap();
        assert_eq!(source.prepared.as_ref().unwrap().work, p + 128);
        drop(source);
        for shift in [Float::with_val(p, 0), Float::with_val(p, 0.5)] {
            let supplied = shifted_norm(&matrix, &shift, 2, p).unwrap();
            let mut bound = BorderedResidualSource::new(&matrix, &shift, p);
            let expected = bordered_residual_reference(
                &matrix, &shift, &unit, &vector, &unit, &zero, &supplied, p, None,
            )
            .unwrap();
            same_point(
                &bound
                    .evaluate(&unit, &vector, &unit, &zero, &supplied)
                    .unwrap(),
                &expected,
            );
        }
    }

    #[test]
    fn prepared_bordered_retention_denial_and_invalid_inputs_preserve_old_errors() {
        let p = 128;
        let (matrix, eigenvalue, unit, forcing, response) = fixture(17, p);
        let zero = Float::with_val(p, 0);
        let supplied = shifted_norm(&matrix, &eigenvalue, 17, p).unwrap();
        let resources = xc_core::RunResources::new(32 << 20).unwrap();
        let _scope = resources.enter();
        let mut source = BorderedResidualSource::new(&matrix, &eigenvalue, p);
        source
            .evaluate(&unit, &forcing, &response, &zero, &supplied)
            .unwrap();
        let retained = resources.usage().retained_bytes;
        assert!(retained > 0);
        let occupied = resources
            .try_reserve_retained(resources.maximum_bytes() - retained - 1)
            .unwrap();
        let expected = bordered_residual_reference(
            &matrix,
            &eigenvalue,
            &unit,
            &forcing,
            &response,
            &zero,
            &supplied,
            p,
            None,
        )
        .unwrap();
        same_point(
            &source
                .evaluate(&unit, &forcing, &response, &zero, &supplied)
                .unwrap(),
            &expected,
        );
        assert!(source.prepared.is_none());
        assert_eq!(resources.usage().retained_bytes, occupied.bytes());
        assert_eq!(resources.usage().working_bytes, 0);
        drop(occupied);
        for slot in 0..7 {
            let mut data = vec![Float::with_val(p, 1); 7];
            data[slot] = Float::with_val(p, Special::Nan);
            data[(slot + 2) % 7] = Float::with_val(1_000_129, 1);
            let expected = bordered_residual_reference(
                &data[0..1],
                &data[1],
                &data[2..3],
                &data[3..4],
                &data[4..5],
                &data[5],
                &data[6],
                p,
                None,
            )
            .unwrap_err()
            .to_string();
            let actual = bordered_residual(
                &data[0..1],
                &data[1],
                &data[2..3],
                &data[3..4],
                &data[4..5],
                &data[5],
                &data[6],
                p,
            )
            .unwrap_err()
            .to_string();
            assert_eq!(actual, expected);
        }
        let tiny = xc_core::RunResources::new(1).unwrap();
        let _tiny_scope = tiny.enter();
        let expected = bordered_residual_reference(
            &matrix,
            &eigenvalue,
            &unit,
            &forcing,
            &response,
            &zero,
            &supplied,
            p,
            None,
        )
        .unwrap_err()
        .to_string();
        assert_eq!(
            source
                .evaluate(&unit, &forcing, &response, &zero, &supplied)
                .unwrap_err()
                .to_string(),
            expected
        );
    }

    #[test]
    fn prepared_bordered_extended_guards_and_refusal_match_the_old_route() {
        let p = 8192;
        let matrix = midpoint_matrix(p);
        let zero = Float::with_val(p, 0);
        let one = Float::with_val(p, 1);
        let unit = [one.clone(), zero.clone()];
        let vector = [zero.clone(), zero.clone()];
        let resources = xc_core::RunResources::new(8 << 20).unwrap();
        let _scope = resources.enter();
        for maximum in [4096, 16384] {
            let policy = GuardPolicy::new(p, maximum).unwrap();
            let mut source = BorderedResidualSource::new(&matrix, &zero, p);
            let expected = bordered_residual_reference(
                &matrix,
                &zero,
                &unit,
                &vector,
                &vector,
                &zero,
                &one,
                p,
                Some(policy),
            );
            let actual =
                source.evaluate_with_policy(&unit, &vector, &vector, &zero, &one, Some(policy));
            match (actual, expected) {
                (Ok(actual), Ok(expected)) => {
                    same_point(&actual, &expected);
                    assert_eq!(source.prepared.as_ref().unwrap().work, p + 16384);
                }
                (Err(actual), Err(expected)) => {
                    assert_eq!(actual.to_string(), expected.to_string())
                }
                _ => panic!("prepared and old guard outcomes differ"),
            }
        }
        let occupied = resources
            .try_reserve_retained(resources.maximum_bytes() - 1)
            .unwrap();
        let policy = GuardPolicy::new(p, 16384).unwrap();
        let expected = bordered_residual_reference(
            &matrix,
            &zero,
            &unit,
            &vector,
            &vector,
            &zero,
            &one,
            p,
            Some(policy),
        )
        .unwrap_err()
        .to_string();
        let mut source = BorderedResidualSource::new(&matrix, &zero, p);
        assert_eq!(
            source
                .evaluate_with_policy(&unit, &vector, &vector, &zero, &one, Some(policy))
                .unwrap_err()
                .to_string(),
            expected
        );
        assert!(expected.contains("memory budget"));
        assert_eq!(resources.usage().retained_bytes, occupied.bytes());
        assert_eq!(resources.usage().working_bytes, 0);
    }

    unsafe extern "C" {
        fn mpfr_set_emin(value: std::os::raw::c_long) -> i32;
    }
    struct RestoreExponentMin(i32);
    impl Drop for RestoreExponentMin {
        fn drop(&mut self) {
            assert_eq!(unsafe { mpfr_set_emin(self.0 as _) }, 0);
        }
    }
    #[test]
    fn prepared_bordered_mixed_worker_ranges_use_the_callers_serial_arithmetic() {
        let normal = rug::float::exp_min();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .start_handler(|index| {
                if index != 0 {
                    assert_eq!(unsafe { mpfr_set_emin(-20) }, 0);
                }
            })
            .build()
            .unwrap();
        pool.install(|| {
            let _restore = RestoreExponentMin(rug::float::exp_min());
            assert_eq!(unsafe { mpfr_set_emin(normal as _) }, 0);
            assert!(xc_numerics::mpfr_interval::ensure_uniform_exponent_range().is_err());
            let (matrix, eigenvalue, unit, mut forcing, mut response) = fixture(17, 128);
            response[0] = Float::with_val(128, 1) >> 100u32;
            forcing[0] = Float::with_val(128, 1) >> 200u32;
            let zero = Float::with_val(128, 0);
            let supplied = shifted_norm(&matrix, &eigenvalue, 17, 128).unwrap();
            let mut source = BorderedResidualSource::new(&matrix, &eigenvalue, 128);
            for minimum in [normal, normal + 1] {
                assert_eq!(unsafe { mpfr_set_emin(minimum as _) }, 0);
                let expected = bordered_residual_reference(
                    &matrix,
                    &eigenvalue,
                    &unit,
                    &forcing,
                    &response,
                    &zero,
                    &supplied,
                    128,
                    None,
                )
                .unwrap();
                same_point(
                    &source
                        .evaluate(&unit, &forcing, &response, &zero, &supplied)
                        .unwrap(),
                    &expected,
                );
                assert_eq!(source.prepared.as_ref().unwrap().exponent_range.0, minimum);
            }
        });
    }

    #[test]
    #[ignore = "bounded before/after residual microbenchmark; run explicitly"]
    fn prepared_bordered_before_after_benchmark() {
        // Bound the higher-precision case to four events per trial. Every
        // prepared trial includes its cold first event and retains one source
        // only for that trial, so the repeats do not hide preparation cost.
        for (n, p, events) in [(65usize, 1024u32, 8usize), (129, 3386, 4)] {
            for threads in [1, 4] {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap()
                    .install(|| {
                        let resources =
                            xc_core::RunResources::with_worker_limit(256 << 20, threads).unwrap();
                        let _scope = resources.enter();
                        let (matrix, eigenvalue, unit, forcing, response) = fixture(n, p);
                        let zero = Float::with_val(p, 0);
                        let supplied = shifted_norm(&matrix, &eigenvalue, n, p).unwrap();
                        // Prepare event inputs outside either timer and give
                        // both routes the same varying stored binary points.
                        let inputs = (0..events)
                            .map(|event| {
                                let mut forcing = forcing.clone();
                                let mut response = response.clone();
                                forcing[event % n] += Float::with_val(
                                    p,
                                    Rational::from((event as i32 + 1, 29)),
                                );
                                response[(3 * event + 1) % n] -= Float::with_val(
                                    p,
                                    Rational::from((event as i32 + 2, 31)),
                                );
                                (forcing, response)
                            })
                            .collect::<Vec<_>>();
                        let mut old_trials = Vec::new();
                        let mut prepared_trials = Vec::new();
                        for trial in 0..3 {
                            let mut reference = || {
                                let started = std::time::Instant::now();
                                let values = inputs
                                    .iter()
                                    .map(|(forcing, response)| {
                                        bordered_residual_reference(
                                            &matrix, &eigenvalue, &unit, forcing, response,
                                            &zero, &supplied, p, None,
                                        )
                                        .unwrap()
                                    })
                                    .collect::<Vec<_>>();
                                old_trials.push(started.elapsed().as_secs_f64());
                                std::hint::black_box(values)
                            };
                            let mut prepared = || {
                                let started = std::time::Instant::now();
                                let mut source =
                                    BorderedResidualSource::new(&matrix, &eigenvalue, p);
                                let values = inputs
                                    .iter()
                                    .map(|(forcing, response)| {
                                        source
                                            .evaluate(&unit, forcing, response, &zero, &supplied)
                                            .unwrap()
                                    })
                                    .collect::<Vec<_>>();
                                drop(source);
                                prepared_trials.push(started.elapsed().as_secs_f64());
                                std::hint::black_box(values)
                            };
                            let (expected, actual) = if trial % 2 == 0 {
                                (reference(), prepared())
                            } else {
                                let actual = prepared();
                                (reference(), actual)
                            };
                            for (actual, expected) in actual.iter().zip(&expected) {
                                same_point(actual, expected);
                            }
                            assert_eq!(resources.usage().retained_bytes, 0);
                            assert_eq!(resources.usage().working_bytes, 0);
                        }
                        let old_seconds = old_trials.iter().sum::<f64>();
                        let prepared_seconds = prepared_trials.iter().sum::<f64>();
                        eprintln!(
                            "{}",
                            serde_json::json!({"dimension":n,"bits":p,"events_per_trial":events,
                                "workers":threads,"trials":3,"old_seconds":old_seconds,
                                "prepared_seconds":prepared_seconds,"old_trial_seconds":old_trials,
                                "prepared_trial_seconds":prepared_trials,"ratio":old_seconds/prepared_seconds,
                                "peak_reserved_bytes":resources.usage().peak_bytes,"bit_identical":true,
                                "scope":"Synthetic residual microbenchmark; not a CCM research measurement."})
                        );
                    });
            }
        }
    }
}
