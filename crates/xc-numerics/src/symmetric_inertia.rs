//! Directed symmetric inertia shared by eigenvector separation checks and certificates.
use crate::interval::IntervalError;
type Result<T> = std::result::Result<T, IntervalError>;
use crate::mpfr_interval::MpfrInterval as I;
use rug::{float::Round, Float};

fn signed(v: &I) -> bool {
    v.validate().is_ok() && (v.is_strictly_positive() || v.upper() < &0)
}
/// Inertia of every symmetric matrix enclosed by the input intervals.
#[derive(Clone, Debug)]
pub enum MpfrInertiaResult {
    Conclusive {
        positive: usize,
        negative: usize,
        pivot_enclosures: Vec<I>,
    },
    Inconclusive {
        pivot_index: usize,
        positive: usize,
        negative: usize,
        zero_or_unresolved: usize,
        pivot_enclosures: Vec<I>,
        reason: String,
    },
}

fn swap(a: &mut [I], n: usize, first: usize, second: usize) {
    if first == second {
        return;
    }
    for j in 0..n {
        a.swap(first * n + j, second * n + j);
    }
    for i in 0..n {
        a.swap(i * n + first, i * n + second);
    }
}
struct Block {
    low: I,
    high: I,
    inverse_11: I,
    inverse_12: I,
    inverse_22: I,
}
fn block(a: &[I], n: usize, first: usize, second: usize, p: u32) -> Result<Option<Block>> {
    let x = &a[first * n + first];
    let y = &a[first * n + second];
    let z = &a[second * n + second];
    let det = x.mul(z).sub(&y.square());
    det.validate()?;
    if det.contains_zero() {
        return Ok(None);
    }
    let root = x
        .sub(z)
        .square()
        .add(&y.square().mul(&I::from_i64(4, p)))
        .sqrt()?;
    let trace = x.add(z);
    let two = I::from_i64(2, p);
    let low = trace.sub(&root).div(&two)?;
    let high = trace.add(&root).div(&two)?;
    if !signed(&low) || !signed(&high) {
        return Ok(None);
    }
    Ok(Some(Block {
        low,
        high,
        inverse_11: z.div(&det)?,
        inverse_12: y.neg().div(&det)?,
        inverse_22: x.div(&det)?,
    }))
}

// Choose the same extrema as four-corner interval multiplication before
// rounding, avoiding products that cannot attain either endpoint.
fn product(a: &I, b: &I) -> Result<I> {
    a.validate()?;
    b.validate()?;
    if a.precision() != b.precision() {
        return Err(IntervalError::Invalid("mixed interval precision".into()));
    }
    let p = a.precision();
    let (al, au, bl, bu) = (a.lower(), a.upper(), b.lower(), b.upper());
    let rounded = |x: &Float, y: &Float, round| Float::with_val_round(p, x * y, round).0;
    let pair = if al >= &0 {
        if bl >= &0 {
            Some((al, bl, au, bu))
        } else if bu <= &0 {
            Some((au, bl, al, bu))
        } else {
            Some((au, bl, au, bu))
        }
    } else if au <= &0 {
        if bl >= &0 {
            Some((al, bu, au, bl))
        } else if bu <= &0 {
            Some((au, bu, al, bl))
        } else {
            Some((al, bu, al, bl))
        }
    } else if bl >= &0 {
        Some((al, bu, au, bu))
    } else if bu <= &0 {
        Some((au, bl, al, bl))
    } else {
        None
    };
    let (lower, upper) = if let Some((ll, lr, ul, ur)) = pair {
        (rounded(ll, lr, Round::Down), rounded(ul, ur, Round::Up))
    } else {
        (
            rounded(al, bu, Round::Down).min(&rounded(au, bl, Round::Down)),
            rounded(al, bl, Round::Up).max(&rounded(au, bu, Round::Up)),
        )
    };
    I::new(lower, upper)
}

/// Outward MPFR arithmetic at fixed precision with exact dyadic pivot records.
/// Congruent symmetric swaps and interval Schur updates preserve enclosure of
/// every admissible symmetric input matrix. Only strictly signed interval
/// pivots contribute to inertia; unresolved pivots remain inconclusive.
pub fn inertia(matrix: &[I], n: usize, p: u32) -> Result<MpfrInertiaResult> {
    inertia_impl(matrix, n, p, false)
}
/// Directed inertia with largest rigorously separated diagonal pivot selection.
/// This avoids dividing by a nearly zero early diagonal when a later stable
/// pivot is available. The historical first-signed route remains separate for
/// bit-exact replay of existing portable MPFR inertia schema 2.
pub fn inertia_stable(matrix: &[I], n: usize, p: u32) -> Result<MpfrInertiaResult> {
    inertia_impl(matrix, n, p, true)
}
fn inertia_impl(matrix: &[I], n: usize, p: u32, stable_pivots: bool) -> Result<MpfrInertiaResult> {
    if !(32..=1_000_000).contains(&p) {
        return Err(IntervalError::Invalid(
            "unsupported interval inertia precision".into(),
        ));
    }
    if n == 0 || n.checked_mul(n) != Some(matrix.len()) {
        return Err(IntervalError::Invalid(
            "invalid inertia matrix shape".into(),
        ));
    }
    for i in 0..n {
        for j in 0..i {
            if matrix[i * n + j].lower() != matrix[j * n + i].lower()
                || matrix[i * n + j].upper() != matrix[j * n + i].upper()
            {
                return Err(IntervalError::Invalid("asymmetric interval matrix".into()));
            }
        }
    }
    for value in matrix {
        value.validate()?;
        if value.precision() != p {
            return Err(IntervalError::Invalid(
                "inertia input precision mismatch".into(),
            ));
        }
    }
    let mut a = matrix.to_vec();
    let mut pivots = Vec::with_capacity(n);
    let mut positive = 0;
    let mut negative = 0;
    let mut k = 0;
    while k < n {
        let diagonal_pivot = if stable_pivots {
            (k..n).filter(|&j| signed(&a[j * n + j])).max_by(|&i, &j| {
                let separation = |index: usize| {
                    let v = &a[index * n + index];
                    if v.is_strictly_positive() {
                        v.lower().clone()
                    } else {
                        -v.upper().clone()
                    }
                };
                separation(i).total_cmp(&separation(j))
            })
        } else {
            (k..n).find(|&j| signed(&a[j * n + j]))
        };
        if let Some(selected) = diagonal_pivot {
            swap(&mut a, n, k, selected);
            let pivot = a[k * n + k].clone();
            pivots.push(pivot.clone());
            if pivot.is_strictly_positive() {
                positive += 1;
            } else {
                negative += 1;
            }
            if k + 1 == n {
                k += 1;
                continue;
            }
            // div() uses this same directed reciprocal before multiplication.
            // Hoist the identical reciprocal once per pivot, preserving all
            // successful interval endpoints while avoiding O(n^3) divisions.
            let inverse_pivot = pivot.reciprocal()?;
            for i in k + 1..n {
                for j in i..n {
                    let numerator = product(&a[i * n + k], &a[j * n + k])?;
                    let correction = product(&numerator, &inverse_pivot)?;
                    let updated = a[i * n + j].sub(&correction);
                    updated.validate()?;
                    a[i * n + j] = updated.clone();
                    a[j * n + i] = updated;
                }
            }
            k += 1;
            continue;
        }
        let mut selected = None;
        'search: for i in k..n {
            for j in i + 1..n {
                if block(&a, n, i, j, p)?.is_some() {
                    selected = Some((i, j));
                    break 'search;
                }
            }
        }
        let Some((i, j)) = selected else {
            pivots.push(a[k * n + k].clone());
            return Ok(MpfrInertiaResult::Inconclusive {
                pivot_index: k,
                positive,
                negative,
                zero_or_unresolved: n - k,
                pivot_enclosures: pivots,
                reason: "no remaining signed MPFR interval 1x1 or 2x2 pivot".into(),
            });
        };
        swap(&mut a, n, k, i);
        let adjusted = if j == k { i } else { j };
        swap(&mut a, n, k + 1, adjusted);
        let b =
            block(&a, n, k, k + 1, p)?.expect("same certified block after symmetric permutation");
        for value in [&b.low, &b.high] {
            pivots.push(value.clone());
            if value.is_strictly_positive() {
                positive += 1;
            } else {
                negative += 1;
            }
        }
        for i in k + 2..n {
            let ri = a[i * n + k].clone();
            let si = a[i * n + k + 1].clone();
            for j in i..n {
                let rj = &a[j * n + k];
                let sj = &a[j * n + k + 1];
                let correction = ri
                    .mul(&b.inverse_11)
                    .mul(rj)
                    .add(&ri.mul(&b.inverse_12).mul(sj))
                    .add(&si.mul(&b.inverse_12).mul(rj))
                    .add(&si.mul(&b.inverse_22).mul(sj));
                let updated = a[i * n + j].sub(&correction);
                updated.validate()?;
                a[i * n + j] = updated.clone();
                a[j * n + i] = updated;
            }
        }
        k += 2;
    }
    Ok(MpfrInertiaResult::Conclusive {
        positive,
        negative,
        pivot_enclosures: pivots,
    })
}

/// Rigorous inertia of the exact stored symmetric matrix minus a stored shift.
/// A conclusive result has no zero eigenvalues; an unresolved pivot does not
/// establish an equality count. The byte limit covers both retained interval
/// matrices, pivot records, and conservative row scratch before allocation.
pub fn point_matrix_inertia_at(
    matrix: &[Float],
    n: usize,
    shift: &Float,
    p: u32,
    maximum_working_bytes: u64,
) -> Result<MpfrInertiaResult> {
    if n == 0
        || n.checked_mul(n) != Some(matrix.len())
        || !(32..=1_000_000).contains(&p)
        || !shift.is_finite()
        || shift.prec() > p
        || matrix.iter().any(|x| !x.is_finite() || x.prec() > p)
    {
        return Err(IntervalError::Invalid(
            "invalid stored inertia matrix, shift, or precision".into(),
        ));
    }
    let scalar_bytes = 64u128 + u128::from(p).div_ceil(8);
    let needed = (4u128 * n as u128 * n as u128 + 16u128 * n as u128) * scalar_bytes;
    if needed > u128::from(maximum_working_bytes) {
        return Err(IntervalError::Inconclusive(
            "stored inertia exceeds explicit working-byte budget".into(),
        ));
    }
    for row in 0..n {
        for column in 0..row {
            if matrix[row * n + column] != matrix[column * n + row] {
                return Err(IntervalError::Invalid(
                    "asymmetric stored inertia matrix".into(),
                ));
            }
        }
    }
    let threshold = I::from_float(shift, p)?;
    let shifted = matrix
        .iter()
        .enumerate()
        .map(|(i, x)| {
            let value = I::from_float(x, p)?;
            let result = if i / n == i % n {
                value.sub(&threshold)
            } else {
                value
            };
            result.validate()?;
            Ok(result)
        })
        .collect::<Result<Vec<_>>>()?;
    inertia_stable(&shifted, n, p)
}

#[cfg(test)]
mod product_tests {
    use super::*;
    #[test]
    fn selected_product_extrema_match_all_four_corners() {
        for p in [32, 128, 256, 9000] {
            for shift in [-5000, 0, 5000] {
                let intervals = [(-7, -3), (-7, 0), (-7, 3), (0, 0), (0, 3), (2, 3)];
                for (al, au) in intervals {
                    for (bl, bu) in intervals {
                        let a = I::new(
                            Float::with_val(p, rug::Rational::from((al, 7)) << shift),
                            Float::with_val(p, rug::Rational::from((au, 7)) << shift),
                        )
                        .unwrap();
                        let b = I::new(
                            Float::with_val(p, rug::Rational::from((bl, 11)) << -shift),
                            Float::with_val(p, rug::Rational::from((bu, 11)) << -shift),
                        )
                        .unwrap();
                        assert_eq!(
                            product(&a, &b).unwrap().to_rational_interval(),
                            a.mul(&b).to_rational_interval()
                        );
                    }
                }
            }
        }
    }
}
