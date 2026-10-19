// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.)
// All rights reserved. See LICENSE in the repository root.

//! Exact products of the binary values stored in MPFR matrices.
//!
//! Each operand is represented as an integer matrix times one power of two.
//! Integer multiply-adds are exact; only the final conversion is rounded.
//! This differs from entrywise MPFR reduction trees and can change their answers.
//! Solvers using it bind the exact-product arithmetic in their cache identity.
//! Exact algebraic zeros are canonical positive zeros, including products of
//! signed input zeros. Nearest means ties to even; Down and Up are directed.
//!
//! The implementation is original, developed with AI assistance, using the
//! elementary dyadic/integer identity and existing Rug/GMP/MPFR dependencies.
//! No external implementation or data is copied. The optional modular adapter
//! delegates to the crate's explicit CPU/CUDA modular-product backend.

use anyhow::{anyhow, ensure, Result};
use rayon::prelude::*;
use rug::{float::Round, Float, Integer};

/// Row-major product of a rows-by-inner and an inner-by-cols matrix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProductShape {
    pub rows: usize,
    pub inner: usize,
    pub cols: usize,
}

/// Explicit admission policy; no environment-dependent defaults are applied.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExactProductLimits {
    /// Ceiling for the conservative allocation estimate in the plan. This
    /// does not replace a process limit for GMP's internal allocation strategy.
    pub max_workspace_bytes: u128,
    /// Number of exact scalar integer multiply-adds, not a bit-complexity bound.
    pub max_scalar_products: u128,
    /// Maximum difference between canonical dyadic exponents within either input.
    pub max_exponent_span: u32,
    /// Bound on each aligned operand and each exact output accumulator.
    pub max_integer_bits: u32,
}

/// Admission facts computed without allocating copies of input significands.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExactProductPlan {
    pub shape: ProductShape,
    pub precision_bits: u32,
    pub left_exponent: i64,
    pub right_exponent: i64,
    pub left_integer_bits: u32,
    pub right_integer_bits: u32,
    pub output_integer_bits: u32,
    pub scalar_products: u128,
    pub estimated_workspace_bytes: u128,
}

/// Validated integer operands, useful as a future modular-backend boundary.
///
/// Getters expose immutable exact integers and a bound on every output. There
/// is intentionally no API accepting an unchecked accelerator result as a
/// correct product. The optional modular adapter uses the crate's checked
/// exact modular-product implementation and the same final rounding boundary.
#[derive(Debug)]
pub struct PreparedProduct {
    plan: ExactProductPlan,
    left: Vec<Integer>,
    right: Vec<Integer>,
    exponent_range: (i32, i32),
}

#[derive(Clone, Copy)]
struct OperandLayout {
    exponent: i64,
    bits: u32,
}

/// A pre-product workspace refusal. Only this allocation-admission failure may
/// trigger output-tile splitting; arithmetic/domain/backend errors must escape.
#[derive(Debug)]
pub(crate) struct WorkspaceLimit {
    stage: &'static str,
    required: u128,
    available: u128,
}
impl std::fmt::Display for WorkspaceLimit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} workspace budget exceeded: requires {}, available {}",
            self.stage, self.required, self.available
        )
    }
}
impl std::error::Error for WorkspaceLimit {}

fn workspace(required: u128, available: u128, stage: &'static str) -> Result<()> {
    if required > available {
        return Err(WorkspaceLimit {
            stage,
            required,
            available,
        }
        .into());
    }
    Ok(())
}

// Rug's borrowed significand is a positive integer, including for negative
// Floats, with possible padding zeros to a whole limb. Removing these zeros
// before alignment avoids allocating proportional to an absolute exponent.
fn canonical_exponent(value: &Float) -> Option<i64> {
    let significand = value.get_significand()?;
    let trailing = significand
        .find_one(0)
        .expect("normal significand is nonzero");
    Some(
        i64::from(value.get_exp().expect("normal value has exponent"))
            - i64::from(significand.significant_bits())
            + i64::from(trailing),
    )
}

fn operand_layout<'a>(
    values: impl Iterator<Item = &'a Float>,
    limits: ExactProductLimits,
) -> Result<OperandLayout> {
    let mut minimum = i64::MAX;
    let mut maximum = i64::MIN;
    let mut largest_top = i64::MIN;
    for value in values {
        ensure!(value.is_finite(), "exact product inputs must be finite");
        if let Some(exponent) = canonical_exponent(value) {
            minimum = minimum.min(exponent);
            maximum = maximum.max(exponent);
            largest_top = largest_top.max(i64::from(value.get_exp().unwrap()));
        }
    }
    if minimum == i64::MAX {
        return Ok(OperandLayout {
            exponent: 0,
            bits: 0,
        });
    }
    ensure!(
        maximum - minimum <= i64::from(limits.max_exponent_span),
        "exact product input exponent span exceeds budget"
    );
    let bits = u32::try_from(largest_top - minimum)
        .map_err(|_| anyhow!("exact product aligned integer width is unsupported"))?;
    ensure!(
        bits <= limits.max_integer_bits,
        "exact product integer width exceeds budget"
    );
    Ok(OperandLayout {
        exponent: minimum,
        bits,
    })
}

fn multiply(a: u128, b: u128) -> Result<u128> {
    a.checked_mul(b)
        .ok_or_else(|| anyhow!("exact product resource calculation overflow"))
}

fn add(a: u128, b: u128) -> Result<u128> {
    a.checked_add(b)
        .ok_or_else(|| anyhow!("exact product resource calculation overflow"))
}

fn integer_bytes(bits: u32) -> u128 {
    // Rounded limbs, object/container allowance, and allocator slack. The
    // overall estimate below also includes temporary GMP arithmetic space.
    u128::from(bits).div_ceil(64) * 8 + 96
}

fn check_vector_capacity<T>(length: usize) -> Result<()> {
    ensure!(
        length
            .checked_mul(std::mem::size_of::<T>())
            .is_some_and(|bytes| bytes <= isize::MAX as usize),
        "exact product vector capacity is unsupported"
    );
    Ok(())
}

fn aligned<'a>(values: impl Iterator<Item = &'a Float>, exponent: i64) -> Result<Vec<Integer>> {
    values
        .map(|value| {
            let Some(significand) = value.get_significand() else {
                return Ok(Integer::new());
            };
            let trailing = significand.find_one(0).unwrap();
            let mut integer = Integer::from(&*significand >> trailing);
            if value.is_sign_negative() {
                integer = -integer;
            }
            let shift = u32::try_from(canonical_exponent(value).unwrap() - exponent)
                .map_err(|_| anyhow!("exact product alignment shift is unsupported"))?;
            integer <<= shift;
            Ok(integer)
        })
        .collect()
}

/// Validate shape, finiteness and budgets before allocating integer operands.
/// Empty axes are allowed; a zero inner dimension produces exact positive zeros.
pub fn prepare(
    left: &[Float],
    right: &[Float],
    shape: ProductShape,
    precision_bits: u32,
    limits: ExactProductLimits,
) -> Result<PreparedProduct> {
    prepare_borrowed(left.iter(), right.iter(), shape, precision_bits, limits)
}

/// Internal strided-input boundary. Scans borrowed Floats before allocating
/// integer operands, so a rejected LU tile never needs cloned input matrices.
pub(crate) fn prepare_borrowed<'a>(
    left: impl Iterator<Item = &'a Float> + Clone,
    right: impl Iterator<Item = &'a Float> + Clone,
    shape: ProductShape,
    precision_bits: u32,
    limits: ExactProductLimits,
) -> Result<PreparedProduct> {
    ensure!(
        (rug::float::prec_min()..=rug::float::prec_max()).contains(&precision_bits),
        "exact product target precision is unsupported"
    );
    let left_len = shape
        .rows
        .checked_mul(shape.inner)
        .ok_or_else(|| anyhow!("exact product left shape overflows"))?;
    let right_len = shape
        .inner
        .checked_mul(shape.cols)
        .ok_or_else(|| anyhow!("exact product right shape overflows"))?;
    let output_len = shape
        .rows
        .checked_mul(shape.cols)
        .ok_or_else(|| anyhow!("exact product output shape overflows"))?;
    ensure!(
        left.clone().count() == left_len && right.clone().count() == right_len,
        "exact product shape mismatch"
    );
    check_vector_capacity::<Integer>(left_len)?;
    check_vector_capacity::<Integer>(right_len)?;
    check_vector_capacity::<Integer>(output_len)?;
    check_vector_capacity::<Float>(output_len)?;
    let scalar_products = multiply(output_len as u128, shape.inner as u128)?;
    ensure!(
        scalar_products <= limits.max_scalar_products,
        "exact product operation budget exceeded"
    );
    let a = operand_layout(left.clone(), limits)?;
    let b = operand_layout(right.clone(), limits)?;
    let output_bits = if a.bits == 0 || b.bits == 0 || shape.inner == 0 {
        0
    } else {
        let sum_bits = usize::BITS - (shape.inner - 1).leading_zeros();
        a.bits
            .checked_add(b.bits)
            .and_then(|bits| bits.checked_add(sum_bits))
            .ok_or_else(|| anyhow!("exact product accumulator width overflows"))?
    };
    ensure!(
        output_bits <= limits.max_integer_bits,
        "exact product accumulator width exceeds budget"
    );

    let operands = add(
        multiply(left_len as u128, integer_bytes(a.bits))?,
        multiply(right_len as u128, integer_bytes(b.bits))?,
    )?;
    let outputs = multiply(
        output_len as u128,
        add(integer_bytes(output_bits), integer_bytes(precision_bits))?,
    )?;
    let scratch = multiply(16, integer_bytes(output_bits.max(a.bits).max(b.bits)))?;
    let estimated_workspace = multiply(4, add(add(operands, outputs)?, scratch)?)?;
    workspace(
        estimated_workspace,
        limits.max_workspace_bytes,
        "exact product",
    )?;
    let exponent_range = (rug::float::exp_min(), rug::float::exp_max());
    // Integer-to-Float conversion takes place before the exact binary shift.
    // Leave a carry bit, and refuse domains where that conversion can range-round.
    ensure!(
        exponent_range.0 <= 1 && i64::from(output_bits) < i64::from(exponent_range.1),
        "exact product intermediate MPFR exponent domain is unsupported"
    );
    let plan = ExactProductPlan {
        shape,
        precision_bits,
        left_exponent: a.exponent,
        right_exponent: b.exponent,
        left_integer_bits: a.bits,
        right_integer_bits: b.bits,
        output_integer_bits: output_bits,
        scalar_products,
        estimated_workspace_bytes: estimated_workspace,
    };
    Ok(PreparedProduct {
        plan,
        left: aligned(left, a.exponent)?,
        right: aligned(right, b.exponent)?,
        exponent_range,
    })
}

impl PreparedProduct {
    /// Retain every bit of the exact product as a Float, for a subsequent
    /// single-rounded Schur subtraction. Re-admits the wider output storage.
    pub fn with_exact_output(mut self, max_workspace_bytes: u128) -> Result<Self> {
        let precision = self.plan.output_integer_bits.max(2);
        ensure!(
            precision <= rug::float::prec_max(),
            "exact output precision unsupported"
        );
        let entries = multiply(self.plan.shape.rows as u128, self.plan.shape.cols as u128)?;
        let old = multiply(4 * entries, integer_bytes(self.plan.precision_bits))?;
        let new = multiply(4 * entries, integer_bytes(precision))?;
        self.plan.estimated_workspace_bytes = add(self.plan.estimated_workspace_bytes - old, new)?;
        workspace(
            self.plan.estimated_workspace_bytes,
            max_workspace_bytes,
            "exact output",
        )?;
        self.plan.precision_bits = precision;
        Ok(self)
    }

    /// Parallel exact integer products with the same output as the serial oracle.
    /// Each output has its own accumulator; scheduling cannot change its value.
    pub fn cpu_product_parallel(&self, round: Round) -> Result<Vec<Float>> {
        self.check_rounding_context(round)?;
        let ProductShape { rows, inner, cols } = self.plan.shape;
        let integers: Vec<Integer> = (0..rows * cols)
            .into_par_iter()
            .map(|index| {
                let (i, j) = (index / cols, index % cols);
                let mut sum = Integer::new();
                for k in 0..inner {
                    sum += &self.left[i * inner + k] * &self.right[k * cols + j];
                }
                sum
            })
            .collect();
        self.round_integer_product(&integers, round)
    }

    pub fn plan(&self) -> ExactProductPlan {
        self.plan
    }

    /// Exact row-major integer left operand; scale is `plan().left_exponent`.
    pub fn left_integers(&self) -> &[Integer] {
        &self.left
    }

    /// Exact row-major integer right operand; scale is `plan().right_exponent`.
    pub fn right_integers(&self) -> &[Integer] {
        &self.right
    }

    /// Serial exact integer oracle. The scale of the result is the sum of the
    /// two operand exponents. No floating arithmetic or Rayon work is performed.
    pub fn cpu_integer_product(&self) -> Vec<Integer> {
        let ProductShape { rows, inner, cols } = self.plan.shape;
        let mut result = Vec::with_capacity(rows * cols);
        for i in 0..rows {
            for j in 0..cols {
                let mut sum = Integer::new();
                for k in 0..inner {
                    sum += &self.left[i * inner + k] * &self.right[k * cols + j];
                }
                result.push(sum);
            }
        }
        result
    }

    /// Exact product followed by one selected rounding of each mathematical sum.
    ///
    /// Nonzero outputs whose exact normalized exponent is below the active MPFR
    /// minimum, or lacks a possible rounding-carry bit at the maximum, are
    /// refused. Underflow/overflow never silently becomes a finite answer.
    /// The supported scale shift is i32; range changes since preparation refuse.
    pub fn cpu_product(&self, round: Round) -> Result<Vec<Float>> {
        self.check_rounding_context(round)?;
        let integers = self.cpu_integer_product();
        self.round_integer_product(&integers, round)
    }

    /// Exact modular product followed by the same one-rounding conversion as
    /// [`Self::cpu_product`]. This method never changes a solver's backend.
    ///
    /// `limits.max_workspace_bytes` is a **combined** budget: this product's
    /// admitted integer/Float workspace plus the binary bridge are subtracted
    /// before the remaining budget is passed to the modular engine. Caller-owned
    /// input Floats are excluded. The underlying estimates do not include every
    /// allocator or vendor CUDA-context allocation; process admission is still
    /// required. Backend errors are returned without a CPU fallback. Empty axes
    /// use the exact CPU zero result without invoking a backend.
    #[cfg(feature = "modular-product")]
    pub fn modular_product(
        &self,
        round: Round,
        backend: crate::modular_product::ProductBackend,
        limits: crate::modular_product::ProductLimits,
    ) -> Result<Vec<Float>> {
        self.check_rounding_context(round)?;
        let remaining = self.modular_remaining_limits(limits)?;
        let ProductShape { rows, inner, cols } = self.plan.shape;
        if rows == 0 || inner == 0 || cols == 0 {
            return self.cpu_product(round);
        }
        ensure!(
            self.plan.scalar_products <= u128::from(limits.max_modular_scalar_products),
            "exact product minimum modular operation budget exceeded"
        );
        ensure!(
            u64::from(self.plan.left_integer_bits) <= limits.max_input_bits
                && u64::from(self.plan.right_integer_bits) <= limits.max_input_bits,
            "exact product modular bridge input width exceeds budget"
        );
        check_vector_capacity::<num_bigint::BigInt>(self.left.len())?;
        check_vector_capacity::<num_bigint::BigInt>(self.right.len())?;
        // All allocation estimates above are checked before building either
        // bridge array. Digits are unsigned base 2^32, least significant first;
        // signs are handled explicitly, never through text or floating values.
        let left: Vec<_> = self.left.iter().map(integer_to_bigint).collect();
        let right: Vec<_> = self.right.iter().map(integer_to_bigint).collect();
        let entries = match crate::modular_product::multiply(
            &left, &right, rows, inner, cols, backend, remaining,
        ) {
            Ok(result) => {
                ensure!(
                    result.rows == rows && result.columns == cols,
                    "exact product modular backend output shape mismatch"
                );
                result.entries
            }
            Err(error) if error.is::<crate::modular_product::ReconstructionCapacity>() => {
                crate::modular_product::multiply_wide_entries(
                    &left, &right, rows, inner, cols, backend, remaining,
                )?
            }
            Err(error) => return Err(error),
        };
        ensure!(
            entries.len() == rows * cols,
            "exact product modular backend output shape mismatch"
        );
        ensure!(
            entries
                .iter()
                .all(|entry| entry.magnitude().bits() <= u64::from(self.plan.output_integer_bits)),
            "exact product modular backend output width exceeds admitted bound"
        );
        let integers: Vec<_> = entries.iter().map(bigint_to_integer).collect();
        self.round_integer_product(&integers, round)
    }

    #[cfg(feature = "modular-product")]
    fn modular_bridge_bytes(&self) -> Result<u128> {
        let ProductShape { rows, inner, cols } = self.plan.shape;
        if rows == 0 || inner == 0 || cols == 0 {
            return Ok(0);
        }
        let inputs = add(
            multiply(
                self.left.len() as u128,
                integer_bytes(self.plan.left_integer_bits),
            )?,
            multiply(
                self.right.len() as u128,
                integer_bytes(self.plan.right_integer_bits),
            )?,
        )?;
        // Both owned BigInt input arrays plus digit-vector construction slack.
        // The Integer output vector/Float output vector are already in plan;
        // BigInt outputs belong to the modular engine's separate remaining cap.
        let widest = self
            .plan
            .left_integer_bits
            .max(self.plan.right_integer_bits)
            .max(self.plan.output_integer_bits);
        let digit_scratch = multiply(2, integer_bytes(widest))?;
        multiply(4, add(inputs, digit_scratch)?)
    }

    #[cfg(feature = "modular-product")]
    fn modular_remaining_limits(
        &self,
        mut limits: crate::modular_product::ProductLimits,
    ) -> Result<crate::modular_product::ProductLimits> {
        let retained_and_bridge = add(
            self.plan.estimated_workspace_bytes,
            self.modular_bridge_bytes()?,
        )?;
        workspace(
            retained_and_bridge,
            u128::from(limits.max_workspace_bytes),
            "exact product combined modular",
        )?;
        // The comparison above proves both the cast and subtraction are safe.
        limits.max_workspace_bytes -= retained_and_bridge as u64;
        Ok(limits)
    }

    fn check_rounding_context(&self, round: Round) -> Result<()> {
        ensure!(
            matches!(round, Round::Nearest | Round::Down | Round::Up),
            "exact product supports only nearest, down, and up rounding"
        );
        ensure!(
            self.exponent_range == (rug::float::exp_min(), rug::float::exp_max()),
            "exact product MPFR exponent range changed after preparation"
        );
        Ok(())
    }

    /// Internal backend boundary: the caller must already have established
    /// that these are the exact integer products, including CRT uniqueness.
    /// Shape and magnitude checks alone do not establish product correctness.
    pub(crate) fn round_integer_product(
        &self,
        integers: &[Integer],
        round: Round,
    ) -> Result<Vec<Float>> {
        self.check_rounding_context(round)?;
        ensure!(
            integers.len() == self.plan.shape.rows * self.plan.shape.cols,
            "exact product backend output shape mismatch"
        );
        ensure!(
            integers
                .iter()
                .all(|integer| integer.significant_bits() <= self.plan.output_integer_bits),
            "exact product backend output width exceeds admitted bound"
        );
        let exponent = self.plan.left_exponent + self.plan.right_exponent;
        integers
            .iter()
            .map(|integer| self.round_integer(integer, exponent, round))
            .collect()
    }

    fn round_integer(&self, integer: &Integer, exponent: i64, round: Round) -> Result<Float> {
        if integer.is_zero() {
            return Ok(Float::with_val(self.plan.precision_bits, 0));
        }
        let normalized = exponent + i64::from(integer.significant_bits());
        ensure!(
            normalized >= i64::from(self.exponent_range.0)
                && normalized < i64::from(self.exponent_range.1),
            "exact product output exponent is outside the supported finite domain"
        );
        let shift = i32::try_from(exponent)
            .map_err(|_| anyhow!("exact product output binary shift is unsupported"))?;
        // Scaling by a power of two commutes with all three rounding modes
        // when both conversions stay strictly inside the checked normal range.
        let mut value = Float::with_val_round(self.plan.precision_bits, integer, round).0;
        value <<= shift;
        ensure!(
            value.is_finite() && !value.is_zero(),
            "exact product output range failure"
        );
        Ok(value)
    }
}

#[cfg(feature = "modular-product")]
fn integer_to_bigint(integer: &Integer) -> num_bigint::BigInt {
    let sign = if integer.is_zero() {
        num_bigint::Sign::NoSign
    } else if integer.is_negative() {
        num_bigint::Sign::Minus
    } else {
        num_bigint::Sign::Plus
    };
    num_bigint::BigInt::new(sign, integer.to_digits::<u32>(rug::integer::Order::Lsf))
}

#[cfg(feature = "modular-product")]
fn bigint_to_integer(value: &num_bigint::BigInt) -> Integer {
    let (sign, digits) = value.to_u32_digits();
    let integer = Integer::from_digits(&digits, rug::integer::Order::Lsf);
    if sign == num_bigint::Sign::Minus {
        -integer
    } else {
        integer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rug::{float::Special, Rational};

    fn limits() -> ExactProductLimits {
        ExactProductLimits {
            max_workspace_bytes: 64 << 20,
            max_scalar_products: 10_000,
            max_exponent_span: 20_000,
            max_integer_bits: 40_000,
        }
    }

    #[test]
    fn borrowed_strided_preparation_matches_owned_operands() {
        let source: Vec<_> = (0..24).map(|i| Float::with_val(128, i - 7)).collect();
        let left = (1..3).flat_map(|row| source[row * 6..row * 6 + 3].iter());
        let right = (0..3).flat_map(|row| source[row * 6 + 3..row * 6 + 5].iter());
        let owned_left: Vec<_> = left.clone().cloned().collect();
        let owned_right: Vec<_> = right.clone().cloned().collect();
        let shape = ProductShape {
            rows: 2,
            inner: 3,
            cols: 2,
        };
        let borrowed = prepare_borrowed(left, right, shape, 73, limits()).unwrap();
        let owned = prepare(&owned_left, &owned_right, shape, 73, limits()).unwrap();
        assert_eq!(borrowed.plan(), owned.plan());
        assert_eq!(borrowed.left_integers(), owned.left_integers());
        assert_eq!(borrowed.right_integers(), owned.right_integers());
        for round in [Round::Nearest, Round::Down, Round::Up] {
            assert_eq!(
                borrowed.cpu_product(round).unwrap(),
                owned.cpu_product(round).unwrap()
            );
        }
    }

    #[test]
    fn only_preproduct_workspace_refusals_have_the_retry_marker() {
        let small = Float::with_val(64, 1) >> 40u32;
        let mut plus = Float::with_val(64, 1);
        plus += &small;
        let mut minus = Float::with_val(64, 1);
        minus -= &small;
        let (left, right) = ([plus], [minus]);
        let shape = ProductShape {
            rows: 1,
            inner: 1,
            cols: 1,
        };
        let initial = prepare(&left, &right, shape, 64, limits()).unwrap();
        let bytes = initial.plan().estimated_workspace_bytes;
        let mut policy = limits();
        policy.max_workspace_bytes = bytes - 1;
        let admission = prepare(&left, &right, shape, 64, policy).unwrap_err();
        assert!(admission.is::<WorkspaceLimit>());
        // Both exact products carry more than one 64-bit limb. Widening the
        // retained output therefore needs additional, independently admitted space.
        let widened = initial.with_exact_output(bytes).unwrap_err();
        assert!(widened.is::<WorkspaceLimit>());
        assert!(widened.to_string().starts_with("exact output workspace"));
        assert!(prepare(&left, &right, shape, 64, limits())
            .unwrap()
            .with_exact_output(limits().max_workspace_bytes)
            .is_ok());

        policy = limits();
        policy.max_scalar_products = 0;
        assert!(!prepare(&left, &right, shape, 64, policy)
            .unwrap_err()
            .is::<WorkspaceLimit>());
        policy = limits();
        policy.max_integer_bits = 1;
        assert!(!prepare(&left, &right, shape, 64, policy)
            .unwrap_err()
            .is::<WorkspaceLimit>());
        assert!(!prepare(&[], &right, shape, 64, limits())
            .unwrap_err()
            .is::<WorkspaceLimit>());
    }

    fn oracle(a: &[Float], b: &[Float], shape: ProductShape) -> Vec<Rational> {
        (0..shape.rows)
            .flat_map(|i| {
                (0..shape.cols).map(move |j| {
                    let mut sum = Rational::new();
                    for k in 0..shape.inner {
                        sum += a[i * shape.inner + k].to_rational().unwrap()
                            * b[k * shape.cols + j].to_rational().unwrap();
                    }
                    sum
                })
            })
            .collect()
    }

    fn compare(a: &[Float], b: &[Float], shape: ProductShape, precision: u32) {
        let expected = oracle(a, b, shape);
        let product = prepare(a, b, shape, precision, limits()).unwrap();
        for round in [Round::Nearest, Round::Down, Round::Up] {
            let actual = product.cpu_product(round).unwrap();
            for (value, exact) in actual.iter().zip(&expected) {
                let reference = Float::with_val_round(precision, exact, round).0;
                assert_eq!(value, &reference);
                assert_eq!(value.prec(), precision);
                if exact.is_zero() {
                    assert!(!value.is_sign_negative());
                }
                if round == Round::Down {
                    assert!(value.to_rational().unwrap() <= *exact);
                } else if round == Round::Up {
                    assert!(value.to_rational().unwrap() >= *exact);
                }
            }
        }
    }

    #[test]
    fn rectangular_mixed_exponents_match_independent_rationals() {
        let shape = ProductShape {
            rows: 3,
            inner: 5,
            cols: 4,
        };
        let values = |n: usize, offset: i32| {
            (0..n)
                .map(|i| {
                    let mut value =
                        Float::with_val(101 + (i % 3) as u32, (i as i32 * 17 + offset) % 31 - 15);
                    value <<= ((i as i32 * 7 + offset) % 43) - 21;
                    value
                })
                .collect::<Vec<_>>()
        };
        compare(&values(15, 4), &values(20, 11), shape, 19);
        for rows in 1..=3 {
            for inner in 1..=4 {
                for cols in 1..=3 {
                    let shape = ProductShape { rows, inner, cols };
                    compare(
                        &values(rows * inner, -9),
                        &values(inner * cols, 23),
                        shape,
                        11,
                    );
                }
            }
        }
    }

    #[test]
    fn thousands_of_bits_of_cancellation_are_exact() {
        let t: Integer = (Integer::from(1) << 3385_u32) + 123;
        let a = vec![
            Float::with_val(3400, &t),
            Float::with_val(3400, -Integer::from(&t - 1)),
        ];
        let b = vec![
            Float::with_val(3400, &t),
            Float::with_val(3400, Integer::from(&t + 1)),
        ];
        let shape = ProductShape {
            rows: 1,
            inner: 2,
            cols: 1,
        };
        let product = prepare(&a, &b, shape, 53, limits()).unwrap();
        for round in [Round::Nearest, Round::Down, Round::Up] {
            assert_eq!(product.cpu_product(round).unwrap()[0], 1);
        }
        compare(&a, &b, shape, 53);
        let tiny_a = a
            .iter()
            .map(|x| x.clone() >> 10_000_i32)
            .collect::<Vec<_>>();
        compare(&tiny_a, &b, shape, 73);
    }

    #[test]
    fn signed_zeros_and_exact_cancellation_have_canonical_positive_zero() {
        let a = vec![
            Float::with_val(80, Special::NegZero),
            Float::with_val(80, -1),
            Float::with_val(80, 1),
        ];
        let b = vec![
            Float::with_val(90, -3),
            Float::with_val(90, 7),
            Float::with_val(90, 7),
        ];
        compare(
            &a,
            &b,
            ProductShape {
                rows: 1,
                inner: 3,
                cols: 1,
            },
            61,
        );
        compare(
            &a[..1],
            &b[..1],
            ProductShape {
                rows: 1,
                inner: 1,
                cols: 1,
            },
            61,
        );
    }

    #[test]
    fn rounding_ties_signs_and_carry_match_rationals() {
        let a = [15, 17, 19, -15, -17, -19].map(|n| Float::with_val(128, n));
        let b = [Float::with_val(128, 1)];
        compare(
            &a,
            &b,
            ProductShape {
                rows: 6,
                inner: 1,
                cols: 1,
            },
            3,
        );
    }

    #[test]
    fn operand_boundary_reconstructs_exact_inputs_without_precision_padding() {
        let a = [
            Float::with_val(3386, -3) << 100_i32,
            Float::with_val(729, 5) >> 80_i32,
        ];
        let b = [Float::with_val(80, 1), Float::with_val(200, -1)];
        let product = prepare(
            &a,
            &b,
            ProductShape {
                rows: 1,
                inner: 2,
                cols: 1,
            },
            90,
            limits(),
        )
        .unwrap();
        assert_eq!(product.plan().left_exponent, -80);
        assert_eq!(product.plan().left_integer_bits, 182);
        for (inputs, integers, exponent) in [
            (
                &a[..],
                product.left_integers(),
                product.plan().left_exponent,
            ),
            (
                &b[..],
                product.right_integers(),
                product.plan().right_exponent,
            ),
        ] {
            for (input, integer) in inputs.iter().zip(integers) {
                let mut exact = Rational::from(integer);
                exact <<= i32::try_from(exponent).unwrap();
                assert_eq!(input.to_rational().unwrap(), exact);
            }
        }
    }

    #[test]
    fn zero_inner_and_empty_output_shapes_are_defined() {
        let product = prepare(
            &[],
            &[],
            ProductShape {
                rows: 2,
                inner: 0,
                cols: 3,
            },
            32,
            limits(),
        )
        .unwrap();
        assert_eq!(product.cpu_product(Round::Down).unwrap().len(), 6);
        assert!(product
            .cpu_product(Round::Down)
            .unwrap()
            .iter()
            .all(|x| x.is_zero() && !x.is_sign_negative()));
        let product = prepare(
            &[],
            &[],
            ProductShape {
                rows: 0,
                inner: 0,
                cols: 4,
            },
            32,
            limits(),
        )
        .unwrap();
        assert!(product.cpu_product(Round::Up).unwrap().is_empty());
    }

    #[test]
    fn shape_precision_and_nonfinite_inputs_refuse() {
        let one = [Float::with_val(32, 1)];
        let shape = ProductShape {
            rows: 1,
            inner: 1,
            cols: 1,
        };
        assert!(prepare(&one, &[], shape, 32, limits())
            .unwrap_err()
            .to_string()
            .contains("shape"));
        assert!(prepare(&one, &one, shape, 0, limits())
            .unwrap_err()
            .to_string()
            .contains("precision"));
        assert!(prepare(
            &[],
            &[],
            ProductShape {
                rows: usize::MAX,
                inner: 2,
                cols: 0
            },
            32,
            limits()
        )
        .is_err());
        assert!(prepare(
            &[],
            &[],
            ProductShape {
                rows: usize::MAX,
                inner: 0,
                cols: 1
            },
            32,
            limits()
        )
        .unwrap_err()
        .to_string()
        .contains("capacity"));
        for special in [Special::Infinity, Special::NegInfinity, Special::Nan] {
            assert!(
                prepare(&[Float::with_val(32, special)], &one, shape, 32, limits())
                    .unwrap_err()
                    .to_string()
                    .contains("finite")
            );
        }
    }

    #[test]
    fn resource_limits_refuse_before_integer_work() {
        let a = [Float::with_val(80, 3), Float::with_val(80, 5) << 200_i32];
        let b = [Float::with_val(80, 1), Float::with_val(80, 1)];
        let shape = ProductShape {
            rows: 1,
            inner: 2,
            cols: 1,
        };
        let mut budget = limits();
        budget.max_scalar_products = 1;
        assert!(prepare(&a, &b, shape, 40, budget)
            .unwrap_err()
            .to_string()
            .contains("operation"));
        budget = limits();
        budget.max_exponent_span = 199;
        assert!(prepare(&a, &b, shape, 40, budget)
            .unwrap_err()
            .to_string()
            .contains("span"));
        budget = limits();
        budget.max_integer_bits = 100;
        assert!(prepare(&a, &b, shape, 40, budget)
            .unwrap_err()
            .to_string()
            .contains("width"));
        let plan = prepare(&a, &b, shape, 40, limits()).unwrap().plan();
        budget = limits();
        budget.max_workspace_bytes = plan.estimated_workspace_bytes - 1;
        assert!(prepare(&a, &b, shape, 40, budget)
            .unwrap_err()
            .to_string()
            .contains("workspace"));
        budget.max_workspace_bytes += 1;
        assert!(prepare(&a, &b, shape, 40, budget).is_ok());
        budget = limits();
        budget.max_integer_bits = plan.output_integer_bits - 1;
        assert!(prepare(&a, &b, shape, 40, budget)
            .unwrap_err()
            .to_string()
            .contains("accumulator"));
    }

    #[test]
    fn extreme_absolute_exponents_do_not_allocate_exponent_sized_integers() {
        let high = Float::with_val(80, 3) << 100_000_000_i32;
        let low = Float::with_val(80, 5) >> 100_000_000_i32;
        let product = prepare(
            &[high],
            &[low],
            ProductShape {
                rows: 1,
                inner: 1,
                cols: 1,
            },
            40,
            limits(),
        )
        .unwrap();
        assert_eq!(product.plan().left_integer_bits, 2);
        assert_eq!(product.plan().right_integer_bits, 3);
        assert_eq!(product.cpu_product(Round::Nearest).unwrap()[0], 15);
    }

    #[test]
    fn output_underflow_overflow_and_unsupported_rounding_refuse() {
        let shape = ProductShape {
            rows: 1,
            inner: 1,
            cols: 1,
        };
        let high = Float::with_val(80, Float::u_exp(1, rug::float::exp_max() / 2 + 2));
        let low = Float::with_val(80, Float::u_exp(1, rug::float::exp_min() / 2 - 2));
        assert!(high.is_finite() && !high.is_zero() && low.is_finite() && !low.is_zero());
        for input in [high, low] {
            let product = prepare(
                std::slice::from_ref(&input),
                std::slice::from_ref(&input),
                shape,
                40,
                limits(),
            )
            .unwrap();
            for round in [Round::Nearest, Round::Down, Round::Up] {
                assert!(product
                    .cpu_product(round)
                    .unwrap_err()
                    .to_string()
                    .contains("exponent"));
            }
        }
        let one = [Float::with_val(32, 1)];
        let product = prepare(&one, &one, shape, 32, limits()).unwrap();
        assert!(product
            .cpu_product(Round::Zero)
            .unwrap_err()
            .to_string()
            .contains("rounding"));
    }

    #[test]
    fn internal_backend_boundary_checks_shape_width_and_range_context() {
        let one = [Float::with_val(32, 1)];
        let mut product = prepare(
            &one,
            &one,
            ProductShape {
                rows: 1,
                inner: 1,
                cols: 1,
            },
            32,
            limits(),
        )
        .unwrap();
        assert!(product
            .round_integer_product(&[], Round::Nearest)
            .unwrap_err()
            .to_string()
            .contains("shape"));
        assert!(product
            .round_integer_product(&[Integer::from(8)], Round::Nearest)
            .unwrap_err()
            .to_string()
            .contains("width"));
        assert_eq!(
            product
                .round_integer_product(&[Integer::from(1)], Round::Nearest)
                .unwrap()[0],
            1
        );
        product.exponent_range.0 += 1;
        assert!(product
            .cpu_product(Round::Nearest)
            .unwrap_err()
            .to_string()
            .contains("range changed"));
    }

    #[cfg(feature = "modular-product")]
    mod modular_adapter {
        use super::*;
        use crate::modular_product::{ProductBackend, ProductLimits};

        fn modular_limits() -> ProductLimits {
            ProductLimits {
                max_workspace_bytes: 128 << 20,
                max_input_bits: 16_000,
                max_modular_scalar_products: 1_000_000,
                workers: 1,
            }
        }

        fn wide_limb_regression(backend: ProductBackend) {
            let h = Integer::from(1) << 16_000u32;
            let p = 16_010;
            let a = [
                Float::with_val(p, Integer::from(&h + 3)),
                Float::with_val(p, Integer::from(&h - 5)),
                Float::with_val(p, 0),
                -Float::with_val(p, &h),
                Float::with_val(p, Integer::from(&h + 1)),
                Float::with_val(p, 1),
            ];
            let b = [
                Float::with_val(p, Integer::from(&h - 7)),
                Float::with_val(p, 1),
                -Float::with_val(p, Integer::from(&h + 1)),
                Float::with_val(p, -1),
                Float::with_val(p, 13),
                Float::with_val(p, &h),
            ];
            let shape = ProductShape {
                rows: 2,
                inner: 3,
                cols: 2,
            };
            let product = prepare(&a, &b, shape, 101, limits()).unwrap();
            assert!(
                u64::from(product.plan().output_integer_bits)
                    > crate::modular_product::maximum_reconstruction_bits()
            );
            let mut budget = modular_limits();
            budget.max_input_bits = 40_000;
            budget.max_workspace_bytes = 512 << 20;
            let expected = oracle(&a, &b, shape);
            assert_eq!(expected[0], -16);
            for round in [Round::Nearest, Round::Down, Round::Up] {
                let actual = product.modular_product(round, backend, budget).unwrap();
                for (value, rational) in actual.iter().zip(&expected) {
                    let reference = Float::with_val_round(101, rational, round).0;
                    assert_eq!(*value, reference);
                    assert_eq!(value.prec(), reference.prec());
                    assert_eq!(value.is_sign_negative(), reference.is_sign_negative());
                }
            }

            // Rounding the child sums, or the combined product before its Schur
            // subtraction, would erase this nonzero pivot entirely.
            let tiny = Float::with_val(64, 1) >> 24_000u32;
            let mut admission = limits();
            admission.max_exponent_span = 32_000;
            let exact = prepare(
                &[Float::with_val(64, 1), tiny.clone()],
                &[Float::with_val(64, 1), Float::with_val(64, -1)],
                ProductShape {
                    rows: 1,
                    inner: 2,
                    cols: 1,
                },
                64,
                admission,
            )
            .unwrap()
            .with_exact_output(512 << 20)
            .unwrap();
            let wide = exact
                .modular_product(Round::Nearest, backend, budget)
                .unwrap();
            assert_eq!(wide, exact.cpu_product(Round::Nearest).unwrap());
            let mut pivot = Float::with_val(64, 1);
            pivot -= &wide[0];
            assert_eq!(pivot, tiny);
        }

        #[test]
        fn modular_wide_limbs_match_rational_and_preserve_schur_pivot() {
            wide_limb_regression(ProductBackend::Cpu);
        }

        #[test]
        fn modular_near_capacity_keeps_admitted_single_pass_budget() {
            let x = (Integer::from(1) << 11_598u32) + 1;
            let values = [Float::with_val(11_610, &x)];
            let product = prepare(
                &values,
                &values,
                ProductShape {
                    rows: 1,
                    inner: 1,
                    cols: 1,
                },
                101,
                limits(),
            )
            .unwrap();
            assert!(
                u64::from(product.plan().output_integer_bits) + 2
                    > crate::modular_product::maximum_reconstruction_bits()
            );
            let mut budget = modular_limits();
            budget.max_workspace_bytes = 512 << 20;
            budget.max_modular_scalar_products = 2000;
            // The actual bound fits in one pass (at most 1881 moduli), even
            // though the more conservative plan width would suggest limbs.
            assert_eq!(
                product
                    .modular_product(Round::Nearest, ProductBackend::Cpu, budget)
                    .unwrap(),
                product.cpu_product(Round::Nearest).unwrap()
            );
        }

        #[cfg(feature = "cuda-product")]
        #[test]
        #[ignore = "requires an admitted physical CUDA device"]
        fn modular_wide_limbs_cuda_match_rational_and_preserve_schur_pivot() {
            wide_limb_regression(ProductBackend::Cuda);
        }

        fn compare_modular(a: &[Float], b: &[Float], shape: ProductShape, precision: u32) {
            let exact = oracle(a, b, shape);
            let product = prepare(a, b, shape, precision, limits()).unwrap();
            for round in [Round::Nearest, Round::Down, Round::Up] {
                let actual = product
                    .modular_product(round, ProductBackend::Cpu, modular_limits())
                    .unwrap();
                assert_eq!(actual, product.cpu_product(round).unwrap());
                for (value, rational) in actual.iter().zip(&exact) {
                    assert_eq!(value, &Float::with_val_round(precision, rational, round).0);
                    assert_eq!(value.prec(), precision);
                    if rational.is_zero() {
                        assert!(!value.is_sign_negative());
                    }
                    if round == Round::Down {
                        assert!(value.to_rational().unwrap() <= *rational);
                    }
                    if round == Round::Up {
                        assert!(value.to_rational().unwrap() >= *rational);
                    }
                }
            }
        }

        #[test]
        fn modular_3386_bit_cancellation_matches_rational_oracle() {
            let t: Integer = (Integer::from(1) << 3385_u32) + 123;
            let a = [
                Float::with_val(3400, &t),
                Float::with_val(3400, -Integer::from(&t - 1)),
            ];
            let b = [
                Float::with_val(3400, &t),
                Float::with_val(3400, Integer::from(&t + 1)),
            ];
            compare_modular(
                &a,
                &b,
                ProductShape {
                    rows: 1,
                    inner: 2,
                    cols: 1,
                },
                71,
            );
        }

        #[test]
        fn modular_signed_scaling_rectangles_and_directed_bounds_match() {
            let a = [
                Float::with_val(101, -3) << 130_i32,
                Float::with_val(83, 17) >> 91_i32,
                Float::with_val(90, Special::NegZero),
                Float::with_val(88, -19) >> 57_i32,
                Float::with_val(99, 5) << 97_i32,
                Float::with_val(90, 11) >> 30_i32,
            ];
            let b = [
                Float::with_val(77, 13) >> 83_i32,
                Float::with_val(93, -23) << 120_i32,
                Float::with_val(84, -7) << 54_i32,
                Float::with_val(81, 31) >> 110_i32,
                Float::with_val(67, 29) >> 57_i32,
                Float::with_val(96, -37) << 18_i32,
            ];
            compare_modular(
                &a,
                &b,
                ProductShape {
                    rows: 2,
                    inner: 3,
                    cols: 2,
                },
                17,
            );
        }

        #[test]
        fn modular_binary_sign_bridge_is_exact() {
            let large: Integer = (Integer::from(1) << 3385_u32) + 123;
            for integer in [
                Integer::new(),
                Integer::from(1),
                Integer::from(-1),
                large.clone(),
                -large,
            ] {
                let converted = integer_to_bigint(&integer);
                assert_eq!(bigint_to_integer(&converted), integer);
                assert_eq!(
                    converted.sign() == num_bigint::Sign::Minus,
                    integer.is_negative()
                );
            }
        }

        #[test]
        fn modular_combined_budget_is_subtracted_before_bridging() {
            let one = [Float::with_val(64, 3)];
            let product = prepare(
                &one,
                &one,
                ProductShape {
                    rows: 1,
                    inner: 1,
                    cols: 1,
                },
                32,
                limits(),
            )
            .unwrap();
            let charged =
                product.plan().estimated_workspace_bytes + product.modular_bridge_bytes().unwrap();
            assert!(charged > product.plan().estimated_workspace_bytes);
            let mut budget = modular_limits();
            let original_budget = budget.max_workspace_bytes;
            assert_eq!(
                product
                    .modular_remaining_limits(budget)
                    .unwrap()
                    .max_workspace_bytes,
                original_budget - u64::try_from(charged).unwrap()
            );
            budget.max_workspace_bytes = u64::try_from(charged - 1).unwrap();
            assert!(product
                .modular_product(Round::Nearest, ProductBackend::Cpu, budget)
                .unwrap_err()
                .to_string()
                .contains("combined modular workspace"));
            budget = modular_limits();
            budget.max_input_bits = 1;
            assert!(product
                .modular_product(Round::Nearest, ProductBackend::Cpu, budget)
                .unwrap_err()
                .to_string()
                .contains("bridge input width"));
            budget = modular_limits();
            budget.max_modular_scalar_products = 0;
            assert!(product
                .modular_product(Round::Nearest, ProductBackend::Cpu, budget)
                .unwrap_err()
                .to_string()
                .contains("operation budget"));
        }

        #[test]
        fn modular_empty_axes_use_checked_positive_zero_route() {
            for shape in [
                ProductShape {
                    rows: 2,
                    inner: 0,
                    cols: 3,
                },
                ProductShape {
                    rows: 0,
                    inner: 0,
                    cols: 4,
                },
            ] {
                let product = prepare(&[], &[], shape, 32, limits()).unwrap();
                let budget = ProductLimits {
                    max_workspace_bytes: u64::try_from(product.plan().estimated_workspace_bytes)
                        .unwrap(),
                    max_input_bits: 0,
                    max_modular_scalar_products: 0,
                    workers: 0,
                };
                assert_eq!(product.modular_bridge_bytes().unwrap(), 0);
                let result = product
                    .modular_product(Round::Down, ProductBackend::Cpu, budget)
                    .unwrap();
                assert_eq!(result.len(), shape.rows * shape.cols);
                assert!(result.iter().all(|x| x.is_zero() && !x.is_sign_negative()));
            }
        }
    }
}
