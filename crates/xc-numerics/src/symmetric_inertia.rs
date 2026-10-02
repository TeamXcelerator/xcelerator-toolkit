//! Directed symmetric inertia shared by eigenvector separation checks and certificates.
use crate::interval::IntervalError;
type Result<T> = std::result::Result<T, IntervalError>;
use crate::mpfr_interval::MpfrInterval as I;
use rayon::prelude::*;
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

// Symmetric row/column exchange. The input is exchanged as stored, both
// triangles included. Once a Schur update has run, `upper_from` names the
// trailing block whose upper triangle alone is current: exchange exactly the
// upper entries that a full exchange of the reflected block would produce.
// Rows and columns before the block are eliminated and never read again.
fn swap(a: &mut [I], n: usize, first: usize, second: usize, upper_from: Option<usize>) {
    if first == second {
        return;
    }
    let Some(start) = upper_from else {
        for j in 0..n {
            a.swap(first * n + j, second * n + j);
        }
        for i in 0..n {
            a.swap(i * n + first, i * n + second);
        }
        return;
    };
    let (first, second) = (first.min(second), first.max(second));
    let index = |i: usize, j: usize| i.min(j) * n + i.max(j);
    a.swap(first * n + first, second * n + second);
    for j in (start..n).filter(|&j| j != first && j != second) {
        a.swap(index(first, j), index(second, j));
    }
}
// Pivot-column entry (row, column), row > column: the stored input entry
// before any update, then its current upper mirror (column, row).
fn below(a: &[I], n: usize, updated: bool, row: usize, column: usize) -> &I {
    if updated {
        &a[column * n + row]
    } else {
        &a[row * n + column]
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

// Every upper-triangle update in one Schur step reads only the retained pivot
// columns and its own previous entry. Parallel rows preserve each entry's
// operation/rounding order, so both pivot routes, including the historical
// portable schema-2 replay, are bit-identical at any thread count. Only the
// upper triangle of the updated block is written; later steps read it there.
// Scratch is O(n), already covered by point_matrix_inertia_at's row allowance.
fn update_schur_rows(
    a: &mut [I],
    n: usize,
    start: usize,
    p: u32,
    update: impl Fn(usize, usize, &I) -> Result<I> + Sync,
) -> Result<()> {
    let rows = n - start;
    let cells = rows.saturating_mul(rows.saturating_add(1)) / 2;
    let workers = rayon::current_num_threads();
    let exponent_bounds = (rug::float::exp_min(), rug::float::exp_max());
    let parallel =
        workers > 1 && cells.saturating_mul(p as usize) >= workers.saturating_mul(32_768);
    let row_update = |(i, row): (usize, &mut [I])| -> Result<()> {
        // MPFR exponent bounds can be thread-local. Never clone or operate on
        // a stored endpoint under different bounds and then certify it.
        if (rug::float::exp_min(), rug::float::exp_max()) != exponent_bounds {
            return Err(IntervalError::Inconclusive(
                "parallel inertia requires matching MPFR exponent bounds".into(),
            ));
        }
        for (j, entry) in row.iter_mut().enumerate().skip(i) {
            let value = update(i, j, entry)?;
            value.validate()?;
            *entry = value;
        }
        Ok(())
    };
    if parallel {
        a.par_chunks_mut(n)
            .enumerate()
            .skip(start)
            .try_for_each(row_update)?;
    } else {
        for row in a.chunks_mut(n).enumerate().skip(start) {
            row_update(row)?;
        }
    }
    Ok(())
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
    // Rows may run on any worker: confirm every worker's exponent range first,
    // so success never depends on which workers receive rows.
    if rayon::current_num_threads() > 1 {
        crate::mpfr_interval::ensure_uniform_exponent_range()?;
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
    // Set by the first Schur update: from then on only upper entries are current.
    let mut updated = false;
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
            swap(&mut a, n, k, selected, updated.then_some(k));
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
            let column = (k + 1..n)
                .map(|i| below(&a, n, updated, i, k).clone())
                .collect::<Vec<_>>();
            update_schur_rows(&mut a, n, k + 1, p, |i, j, previous| {
                let numerator = product(&column[i - k - 1], &column[j - k - 1])?;
                let correction = product(&numerator, &inverse_pivot)?;
                Ok(previous.sub(&correction))
            })?;
            updated = true;
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
        swap(&mut a, n, k, i, updated.then_some(k));
        let adjusted = if j == k { i } else { j };
        swap(&mut a, n, k + 1, adjusted, updated.then_some(k));
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
        let first = (k + 2..n)
            .map(|i| below(&a, n, updated, i, k).clone())
            .collect::<Vec<_>>();
        let second = (k + 2..n)
            .map(|i| below(&a, n, updated, i, k + 1).clone())
            .collect::<Vec<_>>();
        update_schur_rows(&mut a, n, k + 2, p, |i, j, previous| {
            let ri = &first[i - k - 2];
            let si = &second[i - k - 2];
            let rj = &first[j - k - 2];
            let sj = &second[j - k - 2];
            let correction = ri
                .mul(&b.inverse_11)
                .mul(rj)
                .add(&ri.mul(&b.inverse_12).mul(sj))
                .add(&si.mul(&b.inverse_12).mul(rj))
                .add(&si.mul(&b.inverse_22).mul(sj));
            Ok(previous.sub(&correction))
        })?;
        updated = true;
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
/// Input precision may exceed `p`: outward conversion encloses the exact stored
/// values and shift. A conclusive count applies to every matrix in that enclosure.
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
        || matrix.iter().any(|x| !x.is_finite())
    {
        return Err(IntervalError::Invalid(
            "invalid stored inertia matrix, shift, or precision".into(),
        ));
    }
    let scalar_bytes = 64u128 + u128::from(p).div_ceil(8);
    // Two retained interval matrices plus pivot columns, records, and
    // per-row parallel temporaries. At most n row updates are active.
    let needed = (4u128 * n as u128 * n as u128 + 64u128 * n as u128) * scalar_bytes;
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
    // Frozen implementation preceding upper-triangle-only Schur storage: the
    // full input and a serial reflection after every update.
    fn reference_swap(a: &mut [I], n: usize, first: usize, second: usize) {
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
    fn reference_update_schur_rows(
        a: &mut [I],
        n: usize,
        start: usize,
        p: u32,
        update: impl Fn(usize, usize, &I) -> Result<I> + Sync,
    ) -> Result<()> {
        let rows = n - start;
        let cells = rows.saturating_mul(rows.saturating_add(1)) / 2;
        let workers = rayon::current_num_threads();
        let exponent_bounds = (rug::float::exp_min(), rug::float::exp_max());
        let parallel =
            workers > 1 && cells.saturating_mul(p as usize) >= workers.saturating_mul(32_768);
        let row_update = |(i, row): (usize, &mut [I])| -> Result<()> {
            // MPFR exponent bounds can be thread-local. Never clone or operate on
            // a stored endpoint under different bounds and then certify it.
            if (rug::float::exp_min(), rug::float::exp_max()) != exponent_bounds {
                return Err(IntervalError::Inconclusive(
                    "parallel inertia requires matching MPFR exponent bounds".into(),
                ));
            }
            for (j, entry) in row.iter_mut().enumerate().skip(i) {
                let value = update(i, j, entry)?;
                value.validate()?;
                *entry = value;
            }
            Ok(())
        };
        if parallel {
            a.par_chunks_mut(n)
                .enumerate()
                .skip(start)
                .try_for_each(row_update)?;
        } else {
            for row in a.chunks_mut(n).enumerate().skip(start) {
                row_update(row)?;
            }
        }
        for i in start..n {
            for j in i + 1..n {
                a[j * n + i] = a[i * n + j].clone();
            }
        }
        Ok(())
    }
    fn reference_inertia_impl(
        matrix: &[I],
        n: usize,
        p: u32,
        stable_pivots: bool,
    ) -> Result<MpfrInertiaResult> {
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
        // Rows may run on any worker: confirm every worker's exponent range first,
        // so success never depends on which workers receive rows.
        if rayon::current_num_threads() > 1 {
            crate::mpfr_interval::ensure_uniform_exponent_range()?;
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
                reference_swap(&mut a, n, k, selected);
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
                let column = (k + 1..n).map(|i| a[i * n + k].clone()).collect::<Vec<_>>();
                reference_update_schur_rows(&mut a, n, k + 1, p, |i, j, previous| {
                    let numerator = product(&column[i - k - 1], &column[j - k - 1])?;
                    let correction = product(&numerator, &inverse_pivot)?;
                    Ok(previous.sub(&correction))
                })?;
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
            reference_swap(&mut a, n, k, i);
            let adjusted = if j == k { i } else { j };
            reference_swap(&mut a, n, k + 1, adjusted);
            let b = block(&a, n, k, k + 1, p)?
                .expect("same certified block after symmetric permutation");
            for value in [&b.low, &b.high] {
                pivots.push(value.clone());
                if value.is_strictly_positive() {
                    positive += 1;
                } else {
                    negative += 1;
                }
            }
            let first = (k + 2..n).map(|i| a[i * n + k].clone()).collect::<Vec<_>>();
            let second = (k + 2..n)
                .map(|i| a[i * n + k + 1].clone())
                .collect::<Vec<_>>();
            reference_update_schur_rows(&mut a, n, k + 2, p, |i, j, previous| {
                let ri = &first[i - k - 2];
                let si = &second[i - k - 2];
                let rj = &first[j - k - 2];
                let sj = &second[j - k - 2];
                let correction = ri
                    .mul(&b.inverse_11)
                    .mul(rj)
                    .add(&ri.mul(&b.inverse_12).mul(sj))
                    .add(&si.mul(&b.inverse_12).mul(rj))
                    .add(&si.mul(&b.inverse_22).mul(sj));
                Ok(previous.sub(&correction))
            })?;
            k += 2;
        }
        Ok(MpfrInertiaResult::Conclusive {
            positive,
            negative,
            pivot_enclosures: pivots,
        })
    }

    fn same_interval(a: &I, b: &I) -> bool {
        let same = |x: &Float, y: &Float| {
            x.prec() == y.prec()
                && if x.is_nan() || y.is_nan() {
                    x.is_nan() && y.is_nan()
                } else {
                    x.total_cmp(y).is_eq()
                }
        };
        same(a.lower(), b.lower()) && same(a.upper(), b.upper())
    }

    fn same_outcome(a: &Result<MpfrInertiaResult>, b: &Result<MpfrInertiaResult>) -> bool {
        let pivots = |x: &[I], y: &[I]| {
            x.len() == y.len() && x.iter().zip(y).all(|(x, y)| same_interval(x, y))
        };
        match (a, b) {
            (
                Ok(MpfrInertiaResult::Conclusive {
                    positive: p1,
                    negative: n1,
                    pivot_enclosures: e1,
                }),
                Ok(MpfrInertiaResult::Conclusive {
                    positive: p2,
                    negative: n2,
                    pivot_enclosures: e2,
                }),
            ) => p1 == p2 && n1 == n2 && pivots(e1, e2),
            (
                Ok(MpfrInertiaResult::Inconclusive {
                    pivot_index: i1,
                    positive: p1,
                    negative: n1,
                    zero_or_unresolved: z1,
                    pivot_enclosures: e1,
                    reason: r1,
                }),
                Ok(MpfrInertiaResult::Inconclusive {
                    pivot_index: i2,
                    positive: p2,
                    negative: n2,
                    zero_or_unresolved: z2,
                    pivot_enclosures: e2,
                    reason: r2,
                }),
            ) => i1 == i2 && p1 == p2 && n1 == n2 && z1 == z2 && r1 == r2 && pivots(e1, e2),
            (Err(x), Err(y)) => format!("{x:?}") == format!("{y:?}"),
            _ => false,
        }
    }

    // Symmetric fixtures exercising 1x1 and 2x2 pivots, nontrivial swaps,
    // interval entries, unresolved pivots, and lower-triangle zeros whose sign
    // differs from the upper mirror (equal values, so admitted as symmetric).
    fn schur_fixture(kind: usize, n: usize, p: u32) -> Vec<I> {
        let mut state = (kind * 7919 + n * 104_729 + p as usize) as u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as i64
        };
        let mut a = vec![I::from_i64(0, p); n * n];
        for i in 0..n {
            for j in i..n {
                let value = match kind {
                    // Diagonally dominant with scattered weak diagonals.
                    0 => {
                        if i == j {
                            if next() % 3 == 0 {
                                I::from_i64(next() % 5 - 2, p)
                            } else {
                                I::from_i64(4 * n as i64 + next() % 9, p)
                            }
                        } else {
                            I::from_i64(next() % 7 - 3, p)
                                .div(&I::from_i64(next() % 5 + 3, p))
                                .unwrap()
                        }
                    }
                    // Zero diagonal: 2x2 pivots chosen after searches and swaps.
                    1 => {
                        if i == j || next() % 4 == 0 {
                            I::from_i64(0, p)
                        } else {
                            I::from_i64(next() % 9 - 4, p)
                        }
                    }
                    // Interval entries around an indefinite matrix.
                    2 => {
                        let center = I::from_i64(
                            if i == j {
                                (next() % 2 * 2 - 1) * (n as i64 + 3)
                            } else {
                                next() % 5 - 2
                            },
                            p,
                        );
                        let radius = Float::with_val(p, 1) >> (p / 2 + (next() % 16) as u32);
                        I::new(
                            Float::with_val_round(p, center.lower() - &radius, Round::Down).0,
                            Float::with_val_round(p, center.upper() + &radius, Round::Up).0,
                        )
                        .unwrap()
                    }
                    // Mostly zero: unresolved pivots end inconclusive.
                    _ => {
                        if next() % 5 == 0 {
                            I::from_i64(next() % 3 - 1, p)
                        } else {
                            I::from_i64(0, p)
                        }
                    }
                };
                a[i * n + j] = value.clone();
                a[j * n + i] = value;
            }
        }
        for i in 0..n {
            for j in 0..i {
                if a[j * n + i].lower().is_zero()
                    && a[j * n + i].upper().is_zero()
                    && next() % 2 == 0
                {
                    a[i * n + j] = I::point(-Float::with_val(p, 0));
                }
            }
        }
        a
    }

    #[test]
    fn upper_triangle_schur_storage_matches_reflected_reference_bitwise() {
        let mut outcomes = [0usize; 3];
        for p in [64, 128, 512] {
            for kind in 0..4 {
                for n in [1, 2, 3, 5, 8, 17, 40, 64] {
                    let matrix = schur_fixture(kind, n, p);
                    for stable in [true, false] {
                        let expected = rayon::ThreadPoolBuilder::new()
                            .num_threads(1)
                            .build()
                            .unwrap()
                            .install(|| reference_inertia_impl(&matrix, n, p, stable));
                        match &expected {
                            Ok(MpfrInertiaResult::Conclusive { .. }) => outcomes[0] += 1,
                            Ok(MpfrInertiaResult::Inconclusive { .. }) => outcomes[1] += 1,
                            Err(_) => outcomes[2] += 1,
                        }
                        for threads in [1, 2, 4, 8] {
                            let pool = rayon::ThreadPoolBuilder::new()
                                .num_threads(threads)
                                .build()
                                .unwrap();
                            for _ in 0..if n >= 40 { 3 } else { 10 } {
                                let actual = pool.install(|| inertia_impl(&matrix, n, p, stable));
                                assert!(
                                    same_outcome(&actual, &expected),
                                    "p={p} kind={kind} n={n} stable={stable} threads={threads}"
                                );
                            }
                        }
                    }
                }
            }
        }
        // Every outcome class is exercised.
        assert!(outcomes[0] > 0 && outcomes[1] > 0, "{outcomes:?}");
    }
    #[test]
    fn schur_rows_are_bit_identical_across_thread_counts() {
        for (p, stable) in [(128, true), (512, true), (128, false), (512, false)] {
            for blocks in [false, true] {
                let n = 64;
                let matrix = (0..n * n)
                    .map(|index| {
                        let (i, j) = (index / n, index % n);
                        I::from_i64(
                            if blocks {
                                if i / 2 == j / 2 && i != j {
                                    1
                                } else {
                                    0
                                }
                            } else if i == j {
                                200 + i as i64
                            } else {
                                ((i + j) % 7) as i64 - 3
                            },
                            p,
                        )
                    })
                    .collect::<Vec<_>>();
                let run = |threads| {
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(threads)
                        .build()
                        .unwrap()
                        .install(|| {
                            let MpfrInertiaResult::Conclusive {
                                positive,
                                negative,
                                pivot_enclosures,
                            } = if stable {
                                inertia_stable(&matrix, n, p)
                            } else {
                                inertia(&matrix, n, p)
                            }
                            .unwrap()
                            else {
                                panic!("known nonsingular fixture unresolved")
                            };
                            (
                                positive,
                                negative,
                                pivot_enclosures
                                    .iter()
                                    .map(I::to_rational_interval)
                                    .collect::<Vec<_>>(),
                            )
                        })
                };
                let serial = run(1);
                for _ in 0..10 {
                    assert_eq!(serial, run(4));
                }
                assert_eq!(
                    (serial.0, serial.1),
                    if blocks { (32, 32) } else { (64, 0) }
                );
                // Bitwise, including signed zeros, against the reflected
                // full-storage reference at several pool sizes.
                let expected = reference_inertia_impl(&matrix, n, p, stable);
                for threads in [1, 2, 4, 8] {
                    let actual = rayon::ThreadPoolBuilder::new()
                        .num_threads(threads)
                        .build()
                        .unwrap()
                        .install(|| inertia_impl(&matrix, n, p, stable));
                    assert!(same_outcome(&actual, &expected));
                }
            }
        }
    }
    #[test]
    fn inertia_precision_reduction_encloses_exact_stored_boundaries() {
        let high = 512;
        let shift = Float::with_val(high, 1) + (Float::with_val(high, 1) >> 400);
        let matrix = [Float::with_val(high, 1)];
        assert!(matches!(
            point_matrix_inertia_at(&matrix, 1, &shift, 128, 1 << 20).unwrap(),
            MpfrInertiaResult::Inconclusive { .. }
        ));
        assert!(matches!(
            point_matrix_inertia_at(&matrix, 1, &shift, high, 1 << 20).unwrap(),
            MpfrInertiaResult::Conclusive {
                negative: 1,
                positive: 0,
                ..
            }
        ));
        let above = [shift];
        let boundary = Float::with_val(high, 1);
        assert!(matches!(
            point_matrix_inertia_at(&above, 1, &boundary, 128, 1 << 20).unwrap(),
            MpfrInertiaResult::Inconclusive { .. }
        ));
        assert!(matches!(
            point_matrix_inertia_at(&above, 1, &boundary, high, 1 << 20).unwrap(),
            MpfrInertiaResult::Conclusive {
                negative: 0,
                positive: 1,
                ..
            }
        ));
    }
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
