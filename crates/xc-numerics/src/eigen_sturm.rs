//! Directed determinant signs for the exact stored MPFR tridiagonal matrix.
//!
//! The recurrence is P0=1, Pj=(d[j-1]-t)P[j-1]-e[j-2]^2 P[j-2].
//! Every conversion and operation encloses its exact real result. Only definite
//! signs and proven exact zeros are used; an interval straddling zero cannot
//! silently become a zero pivot. Exact zero couplings split independent blocks.
//! The same Sturm zero-skipping convention as the native exact-integer route
//! gives strict-below counts. This does not enclose uncertainty in matrix assembly.

use crate::mpfr_interval::MpfrInterval;
use anyhow::{anyhow, Result};
use rug::Float;

fn enclose(value: &Float, precision: u32) -> MpfrInterval {
    // count() starts above every input's stored precision. This conversion
    // is exact, including the threshold; it cannot discard source bits.
    MpfrInterval::point(Float::with_val(precision, value))
}

fn finite(value: &MpfrInterval) -> Result<()> {
    if !value.lower().is_finite() || !value.upper().is_finite() {
        return Err(anyhow!(
            "HP interval Sturm determinant exceeds the finite exponent range"
        ));
    }
    Ok(())
}

fn sign(value: &MpfrInterval) -> Result<i8> {
    finite(value)?;
    if value.lower() > &0 {
        Ok(1)
    } else if value.upper() < &0 {
        Ok(-1)
    } else if value.lower().is_zero() && value.upper().is_zero() {
        Ok(0)
    } else {
        Err(anyhow!(
            "HP interval Sturm determinant sign is unresolved; more precision is required"
        ))
    }
}

fn at_precision(
    diagonal: &[Float],
    off_diagonal: &[Float],
    threshold: &Float,
    precision: u32,
) -> Result<usize> {
    let threshold = enclose(threshold, precision);
    let mut changes = 0;
    let mut start = 0;
    while start < diagonal.len() {
        let mut end = start + 1;
        while end < diagonal.len() && !off_diagonal[end - 1].is_zero() {
            end += 1;
        }
        let mut previous = MpfrInterval::from_i64(1, precision);
        let mut current = enclose(&diagonal[start], precision).sub(&threshold);
        let mut previous_nonzero_sign = 1;
        let mut record = |value: &MpfrInterval| -> Result<()> {
            let next_sign = sign(value)?;
            if next_sign != 0 {
                if next_sign != previous_nonzero_sign {
                    changes += 1;
                }
                previous_nonzero_sign = next_sign;
            }
            Ok(())
        };
        record(&current)?;
        for index in start + 1..end {
            let shifted = enclose(&diagonal[index], precision).sub(&threshold);
            let coupling = enclose(&off_diagonal[index - 1], precision).square();
            finite(&shifted)?;
            finite(&coupling)?;
            let leading = shifted.mul(&current);
            let trailing = coupling.mul(&previous);
            finite(&leading)?;
            finite(&trailing)?;
            let next = leading.sub(&trailing);
            record(&next)?;
            previous = current;
            current = next;
        }
        start = end;
    }
    Ok(changes)
}

/// The caller validates nonempty dimensions, finite inputs and requested precision.
pub(super) fn count(
    diagonal: &[Float],
    off_diagonal: &[Float],
    threshold: &Float,
    requested_precision: u32,
) -> Result<usize> {
    let source_precision = diagonal
        .iter()
        .chain(off_diagonal)
        .chain([threshold])
        .map(Float::prec)
        .max()
        .unwrap_or(requested_precision)
        .max(requested_precision);
    let mut precision = source_precision
        .checked_add(32)
        .filter(|&p| p <= rug::float::prec_max().min(i32::MAX as u32))
        .ok_or_else(|| anyhow!("HP interval Sturm guard precision exceeds the supported range"))?;
    // Determinant recurrence interval widths can grow exponentially with
    // dimension even for well-separated eigenvalues. Permit dimension-aware
    // guard bits while retaining finite work and definite-sign-only acceptance.
    let cap = source_precision
        .saturating_add(64)
        .saturating_add(
            u32::try_from(diagonal.len())
                .unwrap_or(u32::MAX)
                .saturating_mul(2),
        )
        .max(precision.saturating_mul(4))
        .min(1_000_000)
        .min(rug::float::prec_max());
    let cap = cap.max(precision);
    let mut last_error;
    loop {
        match at_precision(diagonal, off_diagonal, threshold, precision) {
            Ok(count) => return Ok(count),
            Err(error) => last_error = Some(error),
        }
        if precision >= cap {
            break;
        }
        precision = precision.saturating_mul(2).min(cap);
    }
    Err(last_error
        .expect("at least one interval attempt")
        .context(format!(
        "HP Sturm count is unverified after bounded interval escalation through {precision} bits"
    )))
}
