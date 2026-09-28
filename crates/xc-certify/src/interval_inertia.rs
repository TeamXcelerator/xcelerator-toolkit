//! Certificate adapter over the shared directed MPFR inertia kernel.
use super::exact::IntervalInertiaResult;
use rug::{float::Round, Float};
use xc_numerics::{
    interval::{IntervalError, RationalInterval},
    mpfr_interval::MpfrInterval as I,
    symmetric_inertia::{self, MpfrInertiaResult},
};
pub fn inertia(
    matrix: &[RationalInterval],
    n: usize,
    p: u32,
) -> Result<IntervalInertiaResult, IntervalError> {
    inertia_route(matrix, n, p, false)
}
pub fn inertia_stable(
    matrix: &[RationalInterval],
    n: usize,
    p: u32,
) -> Result<IntervalInertiaResult, IntervalError> {
    inertia_route(matrix, n, p, true)
}
fn inertia_route(
    matrix: &[RationalInterval],
    n: usize,
    p: u32,
    stable: bool,
) -> Result<IntervalInertiaResult, IntervalError> {
    if !(32..=1_000_000).contains(&p) || n == 0 || n.checked_mul(n) != Some(matrix.len()) {
        return Err(IntervalError::Invalid(
            "invalid interval inertia shape or precision".into(),
        ));
    }
    for row in 0..n {
        for column in 0..row {
            if matrix[row * n + column] != matrix[column * n + row] {
                return Err(IntervalError::Invalid(
                    "asymmetric rational inertia matrix".into(),
                ));
            }
        }
    }
    let matrix = matrix
        .iter()
        .map(|v| {
            I::new(
                Float::with_val_round(p, v.lower(), Round::Down).0,
                Float::with_val_round(p, v.upper(), Round::Up).0,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let records = |values: Vec<I>| {
        values
            .into_iter()
            .map(|v| v.to_rational_interval())
            .collect()
    };
    let result = if stable {
        symmetric_inertia::inertia_stable(&matrix, n, p)?
    } else {
        symmetric_inertia::inertia(&matrix, n, p)?
    };
    Ok(match result {
        MpfrInertiaResult::Conclusive {
            positive,
            negative,
            pivot_enclosures,
        } => IntervalInertiaResult::Conclusive {
            positive,
            negative,
            pivot_enclosures: records(pivot_enclosures),
        },
        MpfrInertiaResult::Inconclusive {
            pivot_index,
            positive,
            negative,
            zero_or_unresolved,
            pivot_enclosures,
            reason,
        } => IntervalInertiaResult::Inconclusive {
            pivot_index,
            positive,
            negative,
            zero_or_unresolved,
            pivot_enclosures: records(pivot_enclosures),
            reason,
        },
    })
}
