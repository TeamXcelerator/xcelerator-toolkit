//! Directed finite cluster compression and complement feedback.
//! The numerical column-selection rule does not certify the full declared span.
use crate::ccm::{
    capture_runtime::{CaptureResourcePolicy, Checkpoints, Stage},
    extended_research::{
        missing, put, report, row, save_arithmetic_enclosure, unresolved, ExtendedAnalysis,
        ExtensionOptions, ExternalResearchInputs,
    },
    retained_evidence::{
        finite_math::{abs, narrow, scale},
        point, scalar, RetainedMatrix,
    },
    state_geometry::RetainedState,
};
use anyhow::{bail, Result};
use rayon::prelude::*;
use rug::Float;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use xc_numerics::{mpfr_interval::MpfrInterval as I, prefix::lossless_decimal};
type Values = BTreeMap<String, I>;
struct Basis {
    columns: Vec<Vec<I>>,
    selected: Vec<usize>,
}
struct Factors {
    values: Vec<I>,
    permutation: Vec<usize>,
}
#[derive(Serialize, Deserialize)]
struct SavedFactor {
    precision: u32,
    values: Vec<[String; 2]>,
    permutation: Vec<usize>,
}
#[derive(Serialize, Deserialize)]
struct SavedSolve {
    precision: u32,
    values: Vec<[String; 2]>,
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
fn factor(matrix: &[I], n: usize, p: u32) -> Result<Option<Factors>> {
    let mut a = matrix.to_vec();
    let mut permutation = (0..n).collect::<Vec<_>>();
    for k in 0..n {
        let mut best = None;
        let mut magnitude = Float::with_val(p, 0);
        for i in k..n {
            let candidate = abs(&a[i * n + k])?;
            if candidate.lower() > &magnitude {
                magnitude = candidate.lower().clone();
                best = Some(i);
            }
        }
        let Some(pivot) = best else { return Ok(None) };
        if pivot != k {
            for j in 0..n {
                a.swap(k * n + j, pivot * n + j)
            }
            permutation.swap(k, pivot);
        }
        let divisor = a[k * n + k].clone();
        for i in k + 1..n {
            let ratio = a[i * n + k].div(&divisor)?;
            a[i * n + k] = ratio.clone();
            for j in k + 1..n {
                a[i * n + j] = a[i * n + j].sub(&ratio.mul(&a[k * n + j]));
                a[i * n + j].validate()?;
            }
        }
    }
    Ok(Some(Factors {
        values: a,
        permutation,
    }))
}
fn solve(f: &Factors, rhs: &[I], p: u32) -> Result<Vec<I>> {
    let n = rhs.len();
    let mut x = f
        .permutation
        .iter()
        .map(|i| rhs[*i].clone())
        .collect::<Vec<_>>();
    for i in 0..n {
        for j in 0..i {
            x[i] = x[i].sub(&f.values[i * n + j].mul(&x[j]));
        }
    }
    for i in (0..n).rev() {
        for j in i + 1..n {
            x[i] = x[i].sub(&f.values[i * n + j].mul(&x[j]));
        }
        x[i] = x[i].div(&f.values[i * n + i])?;
        if x[i].precision() != p {
            bail!("cluster solve interval precision mismatch")
        }
    }
    Ok(x)
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
    let mut exponent = None;
    for i in 0..n {
        for j in 0..n {
            let entry = shifted(i, j)?;
            entry.validate()?;
            for e in [entry.lower(), entry.upper()]
                .iter()
                .filter_map(|v| v.get_exp())
            {
                exponent = Some(exponent.map_or(i64::from(e), |old: i64| old.max(i64::from(e))));
            }
        }
    }
    let exponent = exponent.unwrap_or(0);
    let entry = |i: usize, j: usize| -> Result<I> { scale(&shifted(i, j)?, -exponent) };
    let mut du = vec![vec![I::from_i64(0, p); n]; b];
    for (i, matrix_row) in m.entries.chunks_exact(n).enumerate() {
        for (j, _) in matrix_row.iter().enumerate() {
            let a = entry(i, j)?;
            for column in 0..b {
                du[column][i] = du[column][i].add(&a.mul(&u[column][j]));
            }
        }
    }
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
        "cluster-directed-shifted-complement-v1",
        &s.manifest.content_digest,
        &m.manifest.content_digest,
        o,
        input,
        p,
        exponent,
        &basis.selected,
    ))?;
    let saved = checkpoints
        .load::<SavedFactor>("interval-factor")?
        .filter(|f| {
            f.precision == p
                && f.values.len() == n * n
                && f.permutation.len() == n
                && f.permutation
                    .iter()
                    .copied()
                    .collect::<std::collections::BTreeSet<_>>()
                    .into_iter()
                    .eq(0..n)
        });
    let factors = if let Some(f) = saved {
        Some(Factors {
            values: decode(&f.values, p)?,
            permutation: f.permutation,
        })
    } else {
        let _stage = Stage::new(format!("bounded complement factorization at {p} bits"));
        let f = factor(&h, n, p)?;
        if let Some(f) = &f {
            if let Err(error) = checkpoints.save(
                "interval-factor",
                &SavedFactor {
                    precision: p,
                    values: encode(&f.values),
                    permutation: f.permutation.clone(),
                },
            ) {
                eprintln!("cluster factor checkpoint unavailable: {error}");
            }
        }
        f
    };
    let Some(factors) = factors else {
        return Ok(Some(partial(rows)));
    };
    let mut solutions = Vec::with_capacity(b);
    let mut residuals = Vec::with_capacity(b);
    for (column, rhs) in coupling.iter().enumerate() {
        let _stage = Stage::new(format!("bounded complement solve {column} at {p} bits"));
        let key = format!("interval-solve-{column}");
        let saved = checkpoints
            .load::<SavedSolve>(&key)?
            .filter(|v| v.precision == p && v.values.len() == n);
        let y = if let Some(v) = saved {
            decode(&v.values, p)?
        } else {
            let y = solve(&factors, rhs, p)?;
            if let Err(error) = checkpoints.save(
                &key,
                &SavedSolve {
                    precision: p,
                    values: encode(&y),
                },
            ) {
                eprintln!("cluster solve checkpoint unavailable: {error}");
            }
            y
        };
        // This residual concerns the midpoint vector in the scaled system.
        // Feedback itself uses the verified solution intervals above.
        let midpoint = y.iter().map(I::midpoint_point).collect::<Vec<_>>();
        let residual = h
            .chunks_exact(n)
            .zip(rhs)
            .map(|(row, rhs)| dot(row, &midpoint, p).sub(rhs))
            .collect::<Vec<_>>();
        residuals.push(
            norm2(&residual, p)
                .sqrt()?
                .div(&norm2(rhs, p).sqrt()?.add(&I::from_i64(1, p)))?,
        );
        solutions.push(y);
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
        out.reason=Some(if measured.budget_limited{"complement interval workspace budget exceeded; enclosed compressed operator and coupling retained"}else{"complement inverse or feedback arithmetic unresolved within guard limit; enclosed compressed operator and coupling retained"}.into());
    }
    out.convention=format!("{}; unit columns selected by a fixed requested-precision residual threshold; interval orthonormal coordinates; original-point A-EI scaled before compression and complement inversion; solve residual uses the scaled system; selected subspace is not a full-declared-span or spectral-selection certificate",if supplied.is_some(){"supplied reference columns"}else{"original center-oriented source and first four even Fourier columns; not a prolate reference"});
    Ok(out)
}
