//! Directed finite cluster compression and complement feedback.
//! The numerical column-selection rule does not certify the full declared span.
use crate::ccm::{
    capture_runtime::{CaptureResourcePolicy, Checkpoints, Stage},
    extended_research::{
        missing, put, report, row, save_arithmetic_enclosure, unresolved, ExtendedAnalysis,
        ExtensionOptions, ExternalResearchInputs,
    },
    retained_evidence::{
        finite_math::{narrow, scale},
        point, scalar, RetainedMatrix,
    },
    state_geometry::RetainedState,
};
use anyhow::{bail, Result};
use rayon::prelude::*;
use rug::Assign;
use rug::Float;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use xc_numerics::{mpfr_interval::MpfrInterval as I, prefix::lossless_decimal};
type Values = BTreeMap<String, I>;
struct Basis {
    columns: Vec<Vec<I>>,
    selected: Vec<usize>,
}
#[derive(Serialize, Deserialize)]
struct SavedSolutions {
    precision: u32,
    columns: Vec<Vec<[String; 2]>>,
}
struct Measurement {
    values: Values,
    rows: Vec<Values>,
    precision: u32,
    complete: bool,
    budget_limited: bool,
    selected: Vec<usize>,
}
fn dot(a: &[I], b: &[I], p: u32) -> I {
    a.iter()
        .zip(b)
        .fold(I::from_i64(0, p), |sum, (a, b)| sum.add(&a.mul(b)))
}
fn norm2(a: &[I], p: u32) -> I {
    a.iter()
        .fold(I::from_i64(0, p), |sum, a| sum.add(&a.square()))
}
fn binary_vector(raw: &[Float], p: u32) -> Result<Option<Vec<I>>> {
    let Some(e) = raw.iter().filter_map(Float::get_exp).max() else {
        return Ok(None);
    };
    raw.iter()
        .map(|x| scale(&I::from_float(x, p)?, -i64::from(e)))
        .collect::<Result<_>>()
        .map(Some)
}
fn basis(raw: &[Vec<Float>], requested: u32, p: u32) -> Result<Option<Basis>> {
    let threshold = I::from_float(&(Float::with_val(p, 1) >> (requested / 2)), p)?.square();
    let mut columns: Vec<Vec<I>> = Vec::new();
    let mut selected = Vec::new();
    for (index, raw) in raw.iter().enumerate() {
        let Some(mut v) = binary_vector(raw, p)? else {
            continue;
        };
        let norm = norm2(&v, p).sqrt()?;
        for value in &mut v {
            *value = value.div(&norm)?;
        }
        for _ in 0..2 {
            for q in &columns {
                let projection = dot(&v, q, p);
                for (value, q) in v.iter_mut().zip(q) {
                    *value = value.sub(&q.mul(&projection));
                }
            }
        }
        let squared = norm2(&v, p);
        squared.validate()?;
        if squared.upper() <= threshold.lower() {
            continue;
        }
        if squared.lower() <= threshold.upper() {
            return Ok(None);
        }
        let norm = squared.sqrt()?;
        columns.push(
            v.iter()
                .map(|value| Ok(value.div(&norm)?))
                .collect::<Result<_>>()?,
        );
        selected.push(index);
    }
    Ok(Some(Basis { columns, selected }))
}
/// Approximate inverse of the midpoint of `h` by Gauss-Jordan elimination with
/// partial pivoting at `p` bits. It is only a preconditioner: `verified_solve`
/// proves every enclosure. Rows update in parallel, each in its serial
/// operation order. `None` when a pivot is exactly zero.
fn midpoint_inverse(h: &[I], n: usize, p: u32) -> Option<Vec<Vec<Float>>> {
    let mut rows: Vec<Vec<Float>> = (0..n)
        .map(|i| {
            let mut row: Vec<Float> = h[i * n..(i + 1) * n]
                .iter()
                .map(|v| v.midpoint_point().lower().clone())
                .collect();
            row.extend((0..n).map(|j| Float::with_val(p, u32::from(i == j))));
            row
        })
        .collect();
    for k in 0..n {
        let pivot = (k..n).max_by(|&a, &b| {
            rows[a][k]
                .cmp_abs(&rows[b][k])
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
        if rows[pivot][k].is_zero() {
            return None;
        }
        rows.swap(k, pivot);
        let inverse = Float::with_val(p, rows[k][k].recip_ref());
        let pivot_row: Vec<Float> = rows[k]
            .iter()
            .map(|v| Float::with_val(p, v * &inverse))
            .collect();
        rows.par_iter_mut()
            .enumerate()
            .filter(|(i, _)| *i != k)
            .for_each(|(_, row)| {
                if row[k].is_zero() {
                    return;
                }
                let factor = row[k].clone();
                let mut product = Float::new(p);
                for (value, pivot) in row.iter_mut().zip(&pivot_row).skip(k) {
                    product.assign(&factor * pivot);
                    *value -= &product;
                }
            });
        rows[k] = pivot_row;
    }
    Some(rows.into_iter().map(|row| row[n..].to_vec()).collect())
}
/// Add the exact range of point `r` times interval `x` to directed sums.
fn add_product(lower: &mut Float, upper: &mut Float, r: &Float, x: &I, product: &mut Float) {
    use rug::{
        float::Round,
        ops::{AddAssignRound, AssignRound},
    };
    let (low, high) = if r.is_sign_negative() {
        (x.upper(), x.lower())
    } else {
        (x.lower(), x.upper())
    };
    product.assign_round(r * low, Round::Down);
    lower.add_assign_round(&*product, Round::Down);
    product.assign_round(r * high, Round::Up);
    upper.add_assign_round(&*product, Round::Up);
}
fn magnitude(lower: &Float, upper: &Float) -> Float {
    if lower.cmp_abs(upper) == Some(std::cmp::Ordering::Greater) {
        lower.clone().abs()
    } else {
        upper.clone().abs()
    }
}
/// Verified solves of every interval system `h y = rhs` (Rump's method).
/// With R ~ mid(h)^-1, C contains I - R h and z contains R(rhs - h x~). If
/// ||C||_inf < 1, every matrix in `h` is nonsingular, and since
/// y - x~ = R(rhs - A x~) + (I - R A)(y - x~), each solution lies in
/// x~ + z + C [-e, e] with e = ||z||_inf / (1 - ||C||_inf). Rows run in
/// parallel; every directed sum keeps a fixed order. `None` when ||C||_inf is
/// not proven below one, so an unresolvable system fails without elimination.
fn verified_solve(h: &[I], rhs: &[Vec<I>], n: usize, p: u32) -> Result<Option<Vec<Vec<I>>>> {
    use rug::{
        float::Round,
        ops::{AddAssignRound, AssignRound, SubAssignRound},
    };
    xc_numerics::mpfr_interval::ensure_uniform_exponent_range()?;
    let Some(r) = midpoint_inverse(h, n, p) else {
        return Ok(None);
    };
    // Row sums of |I - R h|, rounded up.
    let row_sums = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut product = Float::new(p);
            let mut total = Float::new(p);
            for j in 0..n {
                let mut lower = Float::new(p);
                let mut upper = Float::new(p);
                for k in 0..n {
                    add_product(
                        &mut lower,
                        &mut upper,
                        &r[i][k],
                        &h[k * n + j],
                        &mut product,
                    );
                }
                let delta = Float::with_val(p, u32::from(i == j));
                let mut c_lower = delta.clone();
                c_lower.sub_assign_round(&upper, Round::Down);
                let mut c_upper = delta;
                c_upper.sub_assign_round(&lower, Round::Up);
                total.add_assign_round(magnitude(&c_lower, &c_upper), Round::Up);
            }
            total
        })
        .collect::<Vec<_>>();
    let norm = row_sums.iter().fold(Float::new(p), |a, b| a.max(b));
    if !norm.is_finite() || norm >= 1 {
        return Ok(None);
    }
    let mut contraction = Float::with_val(p, 1);
    contraction.sub_assign_round(&norm, Round::Down);
    let mut solutions = Vec::with_capacity(rhs.len());
    for b in rhs {
        let center = b
            .iter()
            .map(|v| v.midpoint_point().lower().clone())
            .collect::<Vec<_>>();
        let approximate = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut sum = Float::new(p);
                let mut product = Float::new(p);
                for (r, c) in r[i].iter().zip(&center) {
                    product.assign(r * c);
                    sum += &product;
                }
                sum
            })
            .collect::<Vec<_>>();
        // d contains rhs - h x~.
        let defect = (0..n)
            .into_par_iter()
            .map(|i| -> Result<I> {
                let mut product = Float::new(p);
                let mut lower = Float::new(p);
                let mut upper = Float::new(p);
                for (k, x) in approximate.iter().enumerate() {
                    add_product(&mut lower, &mut upper, x, &h[i * n + k], &mut product);
                }
                let mut d_lower = b[i].lower().clone();
                d_lower.sub_assign_round(&upper, Round::Down);
                let mut d_upper = b[i].upper().clone();
                d_upper.sub_assign_round(&lower, Round::Up);
                Ok(I::new(d_lower, d_upper)?)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        let z = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut product = Float::new(p);
                let mut lower = Float::new(p);
                let mut upper = Float::new(p);
                for (r, d) in r[i].iter().zip(&defect) {
                    add_product(&mut lower, &mut upper, r, d, &mut product);
                }
                (lower, upper)
            })
            .collect::<Vec<_>>();
        let z_norm = z
            .iter()
            .fold(Float::new(p), |a, (l, u)| a.max(&magnitude(l, u)));
        let mut radius = Float::new(p);
        radius.assign_round(&z_norm / &contraction, Round::Up);
        let mut y = Vec::with_capacity(n);
        for (i, (x, (z_lower, z_upper))) in approximate.iter().zip(&z).enumerate() {
            let mut spread = Float::new(p);
            spread.assign_round(&radius * &row_sums[i], Round::Up);
            let mut lower = Float::new(p);
            lower.assign_round(x + z_lower, Round::Down);
            lower.sub_assign_round(&spread, Round::Down);
            let mut upper = Float::new(p);
            upper.assign_round(x + z_upper, Round::Up);
            upper.add_assign_round(&spread, Round::Up);
            let value = I::new(lower, upper)?;
            value.validate()?;
            y.push(value);
        }
        solutions.push(y);
    }
    Ok(Some(solutions))
}
fn encode(values: &[I]) -> Vec<[String; 2]> {
    values
        .iter()
        .map(|v| [lossless_decimal(v.lower()), lossless_decimal(v.upper())])
        .collect()
}
fn decode(values: &[[String; 2]], p: u32) -> Result<Vec<I>> {
    let point = |s: &str| -> Result<Float> {
        let value = Float::with_val(p, Float::parse(s)?);
        if !value.is_finite()
            || (value.is_zero() && xc_core::DecimalLiteral::new(s)?.canonical()?.as_str() != "0")
        {
            bail!("cluster checkpoint endpoint outside exponent range")
        }
        Ok(value)
    };
    values
        .iter()
        .map(|v| Ok(I::new(point(&v[0])?, point(&v[1])?)?))
        .collect()
}
fn narrow_values(values: &Values, requested: u32) -> Result<bool> {
    for value in values.values() {
        value.validate()?;
        let natural = if value.contains_zero() {
            Float::with_val(value.precision(), 1)
        } else {
            value
                .lower()
                .clone()
                .abs()
                .max(&value.upper().clone().abs())
        };
        if !narrow(std::slice::from_ref(value), &natural, requested)? {
            return Ok(false);
        }
    }
    Ok(true)
}
/// Largest binary exponent of the n x n entries (0 when all are zero). Rows
/// are scanned on workers; the exact maximum does not depend on order and the
/// first failure is taken in row-major order.
fn maximum_exponent(
    n: usize,
    shifted: &(impl Fn(usize, usize) -> Result<I> + Sync),
) -> Result<i64> {
    let rows = (0..n)
        .into_par_iter()
        .map(|i| -> Result<Option<i64>> {
            let mut exponent = None;
            for j in 0..n {
                let entry = shifted(i, j)?;
                entry.validate()?;
                for e in [entry.lower(), entry.upper()]
                    .iter()
                    .filter_map(|v| v.get_exp())
                {
                    exponent =
                        Some(exponent.map_or(i64::from(e), |old: i64| old.max(i64::from(e))));
                }
            }
            Ok(exponent)
        })
        .collect::<Vec<_>>();
    let mut exponent = None;
    for row in rows {
        if let Some(e) = row? {
            exponent = Some(exponent.map_or(e, |old: i64| old.max(e)));
        }
    }
    Ok(exponent.unwrap_or(0))
}
/// `du[column][i] = sum_j entry(i, j) * u[column][j]`, each sum in j order on
/// one worker per row; the first failure is taken in row-major order.
fn column_actions(
    n: usize,
    u: &[Vec<I>],
    entry: &(impl Fn(usize, usize) -> Result<I> + Sync),
    p: u32,
) -> Result<Vec<Vec<I>>> {
    let b = u.len();
    let rows = (0..n)
        .into_par_iter()
        .map(|i| -> Result<Vec<I>> {
            let mut row = vec![I::from_i64(0, p); b];
            for j in 0..n {
                let a = entry(i, j)?;
                for (value, column) in row.iter_mut().zip(u) {
                    *value = value.add(&a.mul(&column[j]));
                }
            }
            Ok(row)
        })
        .collect::<Vec<_>>();
    let mut du = vec![Vec::with_capacity(n); b];
    for row in rows {
        for (column, value) in du.iter_mut().zip(row?) {
            column.push(value);
        }
    }
    Ok(du)
}
/// Row residuals `h[i] . y - rhs[i]`, each dot product serial on one worker.
fn row_residuals(h: &[I], rhs: &[I], y: &[I], p: u32) -> Vec<I> {
    h.par_chunks_exact(y.len())
        .zip(rhs)
        .map(|(row, rhs)| dot(row, y, p).sub(rhs))
        .collect()
}
#[allow(clippy::too_many_arguments)] // Source identities and precision/resource policy remain explicit.
fn calculate(
    s: &RetainedState,
    m: &RetainedMatrix<'_>,
    raw: &[Vec<Float>],
    o: &ExtensionOptions,
    input: Option<&ExternalResearchInputs>,
    limit: u64,
    requested: u32,
    p: u32,
) -> Result<Option<Measurement>> {
    let Some(basis) = basis(raw, requested, p)? else {
        return Ok(None);
    };
    let u = basis.columns;
    let b = u.len();
    if b == 0 {
        return Ok(None);
    }
    let n = s.coefficients.len();
    let eigen = I::from_float(&scalar(&s.eigenvalue, s.precision)?, p)?;
    let shifted = |i: usize, j: usize| -> Result<I> {
        let a = I::from_float(&m.entries[i * n + j], p)?;
        Ok(if i == j { a.sub(&eigen) } else { a })
    };
    let exponent = maximum_exponent(n, &shifted)?;
    let entry = |i: usize, j: usize| -> Result<I> { scale(&shifted(i, j)?, -exponent) };
    let du = column_actions(n, &u, &entry, p)?;
    let mut small = vec![I::from_i64(0, p); b * b];
    for a in 0..b {
        for c in a..b {
            let value = dot(&u[a], &du[c], p);
            small[a * b + c] = value.clone();
            small[c * b + a] = value;
        }
    }
    let mut coupling = du.clone();
    for c in 0..b {
        for a in 0..b {
            for i in 0..n {
                coupling[c][i] = coupling[c][i].sub(&u[a][i].mul(&small[a * b + c]));
            }
        }
    }
    let x =
        binary_vector(&s.coefficients, p)?.ok_or_else(|| anyhow::anyhow!("zero cluster source"))?;
    let q = norm2(&x, p);
    let norm = q.sqrt()?;
    let x = x
        .iter()
        .map(|v| Ok(v.div(&norm)?))
        .collect::<Result<Vec<_>>>()?;
    let mut leakage = x.clone();
    for column in &u {
        let overlap = dot(column, &x, p);
        for (value, column) in leakage.iter_mut().zip(column) {
            *value = value.sub(&column.mul(&overlap));
        }
    }
    let vector_workspace =
        (16 * n as u64 * (raw.len() as u64 + 8) + 256) * (u64::from(p).div_ceil(8) + 64);
    let workspace = (n as u64)
        .saturating_mul(n as u64)
        .saturating_mul(12 * (u64::from(p).div_ceil(8) + 64) + 4 * (u64::from(p) / 3 + 128))
        .saturating_add(vector_workspace);
    let mut values = Values::from([
        ("subspace_dimension".into(), I::from_u64(b as u64, p)),
        ("retained_energy_shift".into(), eigen.clone()),
        ("source_leakage_squared".into(), norm2(&leakage, p)),
        (
            "estimated_factorization_workspace_bytes".into(),
            I::from_u64(workspace, p),
        ),
        (
            "column_selection_threshold".into(),
            I::from_float(&(Float::with_val(p, 1) >> (requested / 2)), p)?,
        ),
        (
            "discarded_reference_columns".into(),
            I::from_u64((raw.len() - b) as u64, p),
        ),
    ]);
    for (i, index) in basis.selected.iter().enumerate() {
        values.insert(
            format!("selected_input_column_{i}"),
            I::from_u64(*index as u64, p),
        );
    }
    let mut rows = Vec::with_capacity(b * b);
    for a in 0..b {
        for c in 0..b {
            let mut compressed = scale(&small[a * b + c], exponent)?;
            if a == c {
                compressed = compressed.add(&eigen)
            }
            rows.push(Values::from([
                ("row".into(), I::from_u64(a as u64, p)),
                ("column".into(), I::from_u64(c as u64, p)),
                ("compressed_operator".into(), compressed),
                (
                    "coupling_gram".into(),
                    scale(&dot(&coupling[a], &coupling[c], p), 2 * exponent)?,
                ),
            ]));
        }
    }
    if !narrow_values(&values, requested)? {
        return Ok(None);
    }
    for row in &rows {
        if !narrow_values(row, requested)? {
            return Ok(None);
        }
    }
    let partial = |rows| Measurement {
        values: values.clone(),
        rows,
        precision: p,
        complete: false,
        budget_limited: workspace > limit,
        selected: basis.selected.clone(),
    };
    if workspace > limit {
        return Ok(Some(partial(rows)));
    }
    let mut right = du.clone();
    for a in 0..b {
        for i in 0..n {
            right[a][i] = u[a][i].sub(&du[a][i]);
            for c in 0..b {
                right[a][i] = right[a][i].add(&small[a * b + c].mul(&u[c][i]));
            }
        }
    }
    let mut h = vec![I::from_i64(0, p); n * n];
    h.par_chunks_mut(n)
        .enumerate()
        .try_for_each(|(i, row)| -> Result<()> {
            for (j, value) in row.iter_mut().enumerate() {
                *value = entry(i, j)?;
                for a in 0..b {
                    *value = value
                        .add(&u[a][i].mul(&right[a][j]))
                        .sub(&du[a][i].mul(&u[a][j]));
                }
                value.validate()?;
            }
            Ok(())
        })?;
    let checkpoints = Checkpoints::new(&(
        "cluster-directed-shifted-complement-verified-solve-v2",
        &s.manifest.content_digest,
        &m.manifest.content_digest,
        o,
        input,
        p,
        exponent,
        &basis.selected,
    ))?;
    let saved = checkpoints
        .load::<SavedSolutions>("verified-solutions")?
        .filter(|v| {
            v.precision == p && v.columns.len() == b && v.columns.iter().all(|c| c.len() == n)
        });
    let solutions = if let Some(saved) = saved {
        Some(
            saved
                .columns
                .iter()
                .map(|c| decode(c, p))
                .collect::<Result<Vec<_>>>()?,
        )
    } else {
        let _stage = Stage::new(format!("bounded complement verified solve at {p} bits"));
        let solutions = verified_solve(&h, &coupling, n, p)?;
        if let Some(solutions) = &solutions {
            if let Err(error) = checkpoints.save(
                "verified-solutions",
                &SavedSolutions {
                    precision: p,
                    columns: solutions.iter().map(|y| encode(y)).collect(),
                },
            ) {
                xc_core::progress_message!("cluster solve checkpoint unavailable: {error}");
            }
        }
        solutions
    };
    let Some(solutions) = solutions else {
        return Ok(Some(partial(rows)));
    };
    let mut residuals = Vec::with_capacity(b);
    for (rhs, y) in coupling.iter().zip(&solutions) {
        // This residual concerns the midpoint vector in the scaled system.
        // Feedback itself uses the verified solution intervals above.
        let midpoint = y.iter().map(I::midpoint_point).collect::<Vec<_>>();
        let residual = row_residuals(&h, rhs, &midpoint, p);
        residuals.push(
            norm2(&residual, p)
                .sqrt()?
                .div(&norm2(rhs, p).sqrt()?.add(&I::from_i64(1, p)))?,
        );
    }
    let original_rows = rows.clone();
    for a in 0..b {
        for c in 0..b {
            let feedback = dot(&coupling[a], &solutions[c], p);
            let mut effective = scale(&small[a * b + c].sub(&feedback), exponent)?;
            if a == c {
                effective = effective.add(&eigen)
            }
            let row = &mut rows[a * b + c];
            row.insert(
                "signed_complement_feedback".into(),
                scale(&feedback, exponent)?,
            );
            row.insert("effective_operator".into(), effective);
            row.insert("solve_relative_residual".into(), residuals[c].clone());
            if !narrow_values(row, requested)? {
                return Ok(Some(partial(original_rows)));
            }
        }
    }
    Ok(Some(Measurement {
        values,
        rows,
        precision: p,
        complete: true,
        budget_limited: false,
        selected: basis.selected,
    }))
}
fn save(map: &mut BTreeMap<String, String>, values: Values, p: u32, used: u32) -> Result<()> {
    for (name, value) in values {
        save_arithmetic_enclosure(map, &name, &value, p)?;
    }
    put(map, "arithmetic_precision_bits", &Float::with_val(p, used));
    Ok(())
}
pub(super) fn analyze(
    s: &RetainedState,
    m: Option<&RetainedMatrix<'_>>,
    o: &ExtensionOptions,
    input: Option<&ExternalResearchInputs>,
) -> Result<ExtendedAnalysis> {
    let mut out = report("operator_cluster", s, o);
    let Some(m) = m else {
        return Ok(missing(out, "retained Tau required"));
    };
    // Every parallel MPFR stage below (complement assembly, verified solve,
    // or checkpoint reuse followed by residuals) needs one exponent range.
    xc_numerics::mpfr_interval::ensure_uniform_exponent_range()?;
    let requested = o.working_precision_bits;
    let supplied = input
        .and_then(|i| i.run_once.as_ref())
        .filter(|i| !i.reference_vectors.is_empty());
    let raw = if let Some(run) = supplied {
        run.reference_vectors
            .iter()
            .map(|v| {
                v.iter()
                    .map(|v| scalar(v, input.unwrap().precision_bits))
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        let mut first = s.coefficients.clone();
        if point::orientation(&first, requested) < 0 {
            for x in &mut first {
                *x = -x.clone();
            }
        }
        let mut raw = vec![first];
        for k in 0..s.modes.min(3) + 1 {
            let mut v = vec![Float::with_val(s.precision, 0); s.coefficients.len()];
            v[s.modes - k] = Float::with_val(s.precision, 1);
            v[s.modes + k] = Float::with_val(s.precision, 1);
            raw.push(v);
        }
        raw
    };
    let limit = o
        .maximum_working_bytes
        .unwrap_or(CaptureResourcePolicy::from_environment()?.maximum_working_bytes);
    let scratch = (16 * s.coefficients.len() as u64 * (raw.len() as u64 + 8) + 256)
        * (u64::from(requested + 4096).div_ceil(8) + 64);
    if scratch > limit {
        return Ok(unresolved(
            out,
            "cluster maximum-guard vector scratch exceeds working-byte budget",
        ));
    }
    let mut accepted = None;
    for guard in [64, 128, 256, 512, 1024, 2048, 4096] {
        if let Some(measured) =
            calculate(s, m, &raw, o, input, limit, requested, requested + guard)?
        {
            let done = measured.complete || measured.budget_limited;
            accepted = Some(measured);
            if done {
                break;
            }
        }
    }
    let Some(measured) = accepted else {
        return Ok(unresolved(
            out,
            "cluster rank decision or finite arithmetic unresolved within 4096 guard bits",
        ));
    };
    let b = measured.selected.len();
    save(
        &mut out.values,
        measured.values,
        requested,
        measured.precision,
    )?;
    for (index, values) in measured.rows.into_iter().enumerate() {
        let mut rr = row(index + 1, "cluster_operator_entry");
        save(&mut rr.values, values, requested, measured.precision)?;
        rr.notes.push(format!(
            "orthonormal coordinates from selected input columns {} and {}",
            measured.selected[index / b],
            measured.selected[index % b]
        ));
        if !measured.complete {
            rr.outcome = "unresolved_denominator".into();
        }
        out.rows.push(rr);
    }
    if !measured.complete {
        out.outcome = "partial_unresolved".into();
        out.reason=Some(if measured.budget_limited{"complement interval workspace budget exceeded; enclosed compressed operator and coupling retained"}else{"verified complement solve (||I - R H|| < 1) or feedback arithmetic unresolved within guard limit; enclosed compressed operator and coupling retained"}.into());
    }
    out.convention=format!("{}; unit columns selected by a fixed requested-precision residual threshold; interval orthonormal coordinates; original-point A-EI scaled before compression and complement inversion; complement solutions enclosed by a midpoint-preconditioned verified solve; solve residual uses the scaled system; selected subspace is not a full-declared-span or spectral-selection certificate",if supplied.is_some(){"supplied reference columns"}else{"original center-oriented source and first four even Fourier columns; not a prolate reference"});
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verified_complement_solve_encloses_exact_solutions_identically_across_threads() {
        let (n, p) = (24, 256);
        // Integer system with known integer solutions.
        let a = |i: usize, j: usize| -> i64 {
            if i == j {
                40 + i as i64
            } else {
                ((3 * i + 5 * j) % 13) as i64 - 6
            }
        };
        let matrix = (0..n * n)
            .map(|k| I::from_i64(a(k / n, k % n), p))
            .collect::<Vec<_>>();
        let solutions = (0..4)
            .map(|c| {
                (0..n)
                    .map(|i| ((i * (c + 2)) % 9) as i64 - 4)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let rhs = solutions
            .iter()
            .map(|x| {
                (0..n)
                    .map(|i| I::from_i64((0..n).map(|j| a(i, j) * x[j]).sum(), p))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| verified_solve(&matrix, &rhs, n, p).unwrap().unwrap())
        };
        let endpoints = |y: &[Vec<I>]| {
            y.iter()
                .flatten()
                .map(|v| (v.lower().clone(), v.upper().clone()))
                .collect::<Vec<_>>()
        };
        let serial = run(1);
        for (y, x) in serial.iter().zip(&solutions) {
            for (enclosure, exact) in y.iter().zip(x) {
                assert!(enclosure.lower() <= exact && enclosure.upper() >= exact);
                let width = Float::with_val(p, enclosure.upper() - enclosure.lower());
                assert!(width < Float::with_val(p, 1) >> (p - 32));
            }
        }
        for _ in 0..10 {
            assert_eq!(endpoints(&run(4)), endpoints(&serial));
        }
        // A singular midpoint is reported unresolved rather than enclosed.
        let singular = vec![I::from_i64(1, p); n * n];
        assert!(verified_solve(&singular, &rhs, n, p).unwrap().is_none());
    }

    #[test]
    #[allow(clippy::needless_range_loop)] // The serial reference keeps its original loops.
    fn parallel_cluster_loops_match_serial_reference_at_any_thread_count() {
        let (n, b, p) = (17, 3, 320);
        let value = |seed: usize| {
            let numerator = ((seed * 7919 + 13) % 1009) as i64 - 504;
            I::from_float(&(Float::with_val(p, numerator) >> (seed % 37) as u32), p).unwrap()
        };
        let u = (0..b)
            .map(|c| (0..n).map(|j| value(c * n + j + 5)).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let h = (0..n * n).map(value).collect::<Vec<_>>();
        let y = (0..n)
            .map(|i| value(i + 3).midpoint_point())
            .collect::<Vec<_>>();
        let rhs = (0..n).map(|i| value(i + 11)).collect::<Vec<_>>();
        let bits = |values: &[I]| {
            values
                .iter()
                .map(|v| (v.lower().clone(), v.upper().clone()))
                .collect::<Vec<_>>()
        };
        for failing in [None, Some((5, 9)), Some((0, 0)), Some((n - 1, n - 1))] {
            let shifted = |i: usize, j: usize| -> Result<I> {
                if Some((i, j)) == failing {
                    bail!("entry {i},{j} fails");
                }
                Ok(h[i * n + j].clone())
            };
            // Serial loops as they were before row parallelism.
            let reference = || -> Result<(i64, Vec<Vec<I>>)> {
                let mut exponent = None;
                for i in 0..n {
                    for j in 0..n {
                        let entry = shifted(i, j)?;
                        entry.validate()?;
                        for e in [entry.lower(), entry.upper()]
                            .iter()
                            .filter_map(|v| v.get_exp())
                        {
                            exponent = Some(
                                exponent.map_or(i64::from(e), |old: i64| old.max(i64::from(e))),
                            );
                        }
                    }
                }
                let exponent = exponent.unwrap_or(0);
                let mut du = vec![vec![I::from_i64(0, p); n]; b];
                for i in 0..n {
                    for j in 0..n {
                        let a = scale(&shifted(i, j)?, -exponent)?;
                        for column in 0..b {
                            du[column][i] = du[column][i].add(&a.mul(&u[column][j]));
                        }
                    }
                }
                Ok((exponent, du))
            };
            let expected = reference();
            let expected_residual = h
                .chunks_exact(n)
                .zip(&rhs)
                .map(|(row, rhs)| dot(row, &y, p).sub(rhs))
                .collect::<Vec<_>>();
            for threads in [1, 2, 4, 8] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                for _ in 0..3 {
                    let actual = pool.install(|| -> Result<(i64, Vec<Vec<I>>)> {
                        let exponent = maximum_exponent(n, &shifted)?;
                        let entry =
                            |i: usize, j: usize| -> Result<I> { scale(&shifted(i, j)?, -exponent) };
                        Ok((exponent, column_actions(n, &u, &entry, p)?))
                    });
                    match (&actual, &expected) {
                        (Ok((e1, du1)), Ok((e2, du2))) => {
                            assert_eq!(e1, e2);
                            for (left, right) in du1.iter().zip(du2) {
                                assert_eq!(bits(left), bits(right));
                            }
                        }
                        (Err(left), Err(right)) => assert_eq!(left.to_string(), right.to_string()),
                        _ => panic!("parallel and serial cluster loops disagree on failure"),
                    }
                    let residual = pool.install(|| row_residuals(&h, &rhs, &y, p));
                    assert_eq!(bits(&residual), bits(&expected_residual));
                }
            }
        }
    }
}
