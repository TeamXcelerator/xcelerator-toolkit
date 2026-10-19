//! Finite stored-point quadratic diagnostics with directed arithmetic.
//! Excludes source construction, operator modeling error and ground selection.
use super::{
    coeffs, scalar, ExternalResearchInputs, OperatorComponent, RetainedMatrix, RetainedState,
};
use crate::ccm::retained_evidence::{
    finite_math::{abs, narrow, scale, scale_float},
    point,
};
use anyhow::{bail, Result};
use rug::Float;
use std::collections::BTreeMap;
use xc_numerics::mpfr_interval::MpfrInterval as I;

/// Owned allocations of this evaluator, excluding its borrowed retained Tau.
/// Counts can be tested at campaign dimensions without allocating a matrix.
#[derive(Clone, Copy, Debug, Default)]
struct WorkspaceShape {
    dimension: u64,
    parsed_scalars: u64,
    parsed_vectors: u64,
    components: u64,
    rank_terms: u64,
    largest_literal: u64,
    report_text: u64,
}

fn workspace_overflow() -> anyhow::Error {
    anyhow::anyhow!("arithmetic energy workspace estimate overflow")
}

fn add_count(total: &mut u64, count: usize) -> Result<()> {
    *total = total
        .checked_add(u64::try_from(count).map_err(|_| workspace_overflow())?)
        .ok_or_else(workspace_overflow)?;
    Ok(())
}

impl WorkspaceShape {
    fn literal(&mut self, value: &str) -> Result<()> {
        self.largest_literal = self
            .largest_literal
            .max(u64::try_from(value.len()).map_err(|_| workspace_overflow())?);
        Ok(())
    }

    fn vector(&mut self, values: &[String]) -> Result<()> {
        add_count(&mut self.parsed_vectors, 1)?;
        add_count(&mut self.parsed_scalars, values.len())?;
        for value in values {
            self.literal(value)?;
        }
        Ok(())
    }

    fn from_input(
        dimension: usize,
        eigenvalue: &str,
        cutoff: &str,
        input: &ExternalResearchInputs,
    ) -> Result<Self> {
        let mut shape = Self {
            dimension: u64::try_from(dimension).map_err(|_| workspace_overflow())?,
            ..Self::default()
        };
        shape.literal(eigenvalue)?;
        add_count(&mut shape.report_text, cutoff.len())?;
        if input.components.is_empty() {
            if let Some(run) = &input.run_once {
                add_count(&mut shape.components, run.component_actions.len())?;
                for component in &run.component_actions {
                    shape.vector(&component.action)?;
                    add_count(&mut shape.report_text, component.label.len())?;
                    add_count(&mut shape.report_text, component.convention.len())?;
                }
            }
        } else {
            add_count(&mut shape.components, input.components.len())?;
            for component in &input.components {
                shape.vector(&component.diagonal)?;
                shape.vector(&component.dense)?;
                add_count(&mut shape.report_text, component.label.len())?;
                add_count(&mut shape.rank_terms, component.rank_one.len())?;
                for rank in &component.rank_one {
                    shape.literal(&rank.weight)?;
                    add_count(&mut shape.parsed_scalars, 1)?;
                    shape.vector(&rank.vector)?;
                }
            }
        }
        if let Some(trial) = input
            .target
            .as_ref()
            .and_then(|t| t.trial_coefficients.as_ref())
        {
            shape.vector(trial)?;
        }
        if let Some(deficit) = &input.deficit {
            shape.literal(deficit)?;
        }
        if let Some(kind) = &input.deficit_kind {
            add_count(&mut shape.report_text, kind.len())?;
        }
        Ok(shape)
    }

    fn bytes(self, source_precision: u32, requested: u32) -> Result<u64> {
        let guard_precision = requested.checked_add(4096).ok_or_else(workspace_overflow)?;
        // A Float's limbs plus its header and allocator allowance. Use whole
        // limbs, and four slots per parsed scalar for Vec growth/reallocation.
        let scalar = |p: u32| u128::from(p).div_ceil(64) * 8 + 128;
        let parsed = 4 * u128::from(self.parsed_scalars) * scalar(source_precision)
            + 256
                * (u128::from(self.parsed_vectors)
                    + u128::from(self.components)
                    + u128::from(self.rank_terms)
                    + 1);
        // calculate holds x, ax and combined, plus at most two n-vectors
        // (a rank action, a residual, or q/aq). Twenty-four n-sized interval
        // slots cover these five vectors with fourfold capacity/reallocation
        // allowance and four additional vector lengths. Old and scaled energy
        // vectors coexist: eight slots per component cover both capacities.
        // The fixed 512 slots cover scalar expression/MPFR scratch, map values,
        // scale checks, and formatting temporaries. Retries do not overlap.
        let interval_slots =
            24 * u128::from(self.dimension) + 8 * u128::from(self.components) + 512;
        let intervals = interval_slots * 2 * scalar(guard_precision);
        // Parsing is sequential; allow scratch for the largest source literal,
        // including canonical-zero validation. No decimal expansion of the
        // exponent is made by the compact report formatter.
        let parsing = 64 * u128::from(self.largest_literal);
        // At most eleven measured values plus the precision field, and one
        // energy row per component. Each enclosure has three decimal strings.
        // Allow keys/nodes/row metadata, sixfold JSON escaping of supplied text,
        // and eight simultaneous representations/capacities through managed
        // validation, compact serialization, persistence and research export.
        let digits = (u128::from(requested) * 30103).div_ceil(100000) + 32;
        let report = (u128::from(self.components) + 12) * (3 * (digits + 128) + 1024)
            + 6 * u128::from(self.report_text)
            + 8192;
        u64::try_from(parsed + intervals + parsing + 8 * report).map_err(|_| workspace_overflow())
    }
}

/// Reserve this amount before `measure` parses any owned input arrays, and
/// keep the reservation until the resulting report has been serialized.
pub(super) fn workspace_bytes(
    s: &RetainedState,
    input: &ExternalResearchInputs,
    requested: u32,
) -> Result<u64> {
    WorkspaceShape::from_input(s.coefficients.len(), &s.eigenvalue, &s.cutoff, input)?
        .bytes(input.precision_bits, requested)
}

pub(super) struct Measurement {
    pub values: BTreeMap<String, I>,
    pub energies: Vec<I>,
    pub arithmetic_precision: u32,
    pub zero_trial: bool,
}
pub(super) struct Operator {
    diagonal: Vec<Float>,
    dense: Vec<Float>,
    rank: Vec<(Float, Vec<Float>)>,
}
impl Operator {
    pub(super) fn parse(c: &OperatorComponent, p: u32) -> Result<Self> {
        Ok(Self {
            diagonal: coeffs(&c.diagonal, p)?,
            dense: coeffs(&c.dense, p)?,
            rank: c
                .rank_one
                .iter()
                .map(|r| Ok((scalar(&r.weight, p)?, coeffs(&r.vector, p)?)))
                .collect::<Result<_>>()?,
        })
    }
    pub(super) fn exponent(&self) -> i64 {
        let direct = self
            .diagonal
            .iter()
            .chain(&self.dense)
            .filter_map(Float::get_exp)
            .map(i64::from);
        let rank = self.rank.iter().filter_map(|(w, q)| {
            Some(
                i64::from(w.get_exp()?) + 2 * i64::from(q.iter().filter_map(Float::get_exp).max()?),
            )
        });
        direct.chain(rank).max().unwrap_or(0)
    }
    pub(super) fn action(&self, x: &[I], e: i64, p: u32) -> Result<Vec<I>> {
        let n = x.len();
        let mut out = vec![I::from_i64(0, p); n];
        if !self.dense.is_empty() {
            for (row, value) in self.dense.chunks_exact(n).zip(&mut out) {
                for (a, v) in row.iter().zip(x) {
                    *value = value.add(&stored(a, -e, p)?.mul(v));
                }
            }
        }
        for ((value, a), v) in out.iter_mut().zip(&self.diagonal).zip(x) {
            *value = value.add(&stored(a, -e, p)?.mul(v));
        }
        for (w, q) in &self.rank {
            let qe = i64::from(q.iter().filter_map(Float::get_exp).max().unwrap_or(0));
            let q = q
                .iter()
                .map(|a| stored(a, -qe, p))
                .collect::<Result<Vec<_>>>()?;
            let factor = dot(&q, x, p).mul(&stored(w, 2 * qe - e, p)?);
            for (value, q) in out.iter_mut().zip(q) {
                *value = value.add(&q.mul(&factor));
            }
        }
        Ok(out)
    }
}
pub(super) fn stored(x: &Float, e: i64, p: u32) -> Result<I> {
    Ok(I::from_float(&scale_float(x, e, p)?, p)?)
}
pub(super) fn vector(x: &[Float], p: u32) -> Result<Option<Vec<I>>> {
    let Some(e) = x.iter().filter_map(Float::get_exp).max() else {
        return Ok(None);
    };
    x.iter()
        .map(|x| stored(x, -i64::from(e), p))
        .collect::<Result<Vec<_>>>()
        .map(Some)
}
pub(super) fn dot(x: &[I], y: &[I], p: u32) -> I {
    x.iter()
        .zip(y)
        .fold(I::from_i64(0, p), |a, (x, y)| a.add(&x.mul(y)))
}
pub(super) fn norm2(x: &[I], p: u32) -> I {
    x.iter().fold(I::from_i64(0, p), |a, x| a.add(&x.square()))
}
fn dense_action(a: &[Float], x: &[I], e: i64, p: u32) -> Result<Vec<I>> {
    a.chunks_exact(x.len())
        .map(|row| {
            let mut result = I::from_i64(0, p);
            for (a, x) in row.iter().zip(x) {
                result = result.add(&stored(a, -e, p)?.mul(x));
            }
            Ok(result)
        })
        .collect()
}

#[allow(clippy::too_many_arguments)] // Keep stored input families and output/guard precision explicit.
fn calculate(
    s: &RetainedState,
    m: &RetainedMatrix<'_>,
    input: &ExternalResearchInputs,
    operators: &[Operator],
    compact: &[Vec<Float>],
    trial: Option<&[Float]>,
    e: i64,
    requested: u32,
    p: u32,
) -> Result<Option<Measurement>> {
    let x = vector(&s.coefficients, p)?.ok_or_else(|| anyhow::anyhow!("zero energy source"))?;
    let x2 = norm2(&x, p);
    let norm = x2.sqrt()?;
    let ax = dense_action(&m.entries, &x, e, p)?;
    let total = dot(&x, &ax, p).div(&x2)?;
    let mut energies = Vec::new();
    let mut combined = vec![I::from_i64(0, p); x.len()];
    let residual;
    let sum;
    let closure;
    if !operators.is_empty() {
        for op in operators {
            let a = op.action(&x, e, p)?;
            energies.push(dot(&x, &a, p).div(&x2)?);
            for (sum, a) in combined.iter_mut().zip(a) {
                *sum = sum.add(&a);
            }
        }
        let delta = combined
            .iter()
            .zip(&ax)
            .map(|(a, b)| a.sub(b))
            .collect::<Vec<_>>();
        residual = norm2(&delta, p).sqrt()?.div(&norm)?;
        sum = dot(&x, &combined, p).div(&x2)?;
        closure = dot(&x, &delta, p).div(&x2)?;
    } else {
        let sign = I::from_i64(i64::from(point::orientation(&s.coefficients, p)), p);
        for a in compact {
            let a = a
                .iter()
                .map(|a| stored(a, -e, p))
                .collect::<Result<Vec<_>>>()?;
            energies.push(dot(&x, &a, p).mul(&sign).div(&norm)?);
            for (sum, a) in combined.iter_mut().zip(a) {
                *sum = sum.add(&a);
            }
        }
        let delta = combined
            .iter()
            .zip(&ax)
            .map(|(a, b)| Ok(a.sub(&b.mul(&sign).div(&norm)?)))
            .collect::<Result<Vec<_>>>()?;
        residual = norm2(&delta, p).sqrt()?;
        sum = dot(&x, &combined, p).mul(&sign).div(&norm)?;
        closure = dot(&x, &delta, p).mul(&sign).div(&norm)?;
    }
    let mut absolute = I::from_i64(0, p);
    for energy in &energies {
        absolute = absolute.add(&abs(energy)?);
    }
    let mut values = BTreeMap::new();
    for (name, value) in [
        ("total_tau_energy", total.clone()),
        ("sum_component_energy", sum.clone()),
        ("sum_absolute_component_energy", absolute),
        ("energy_closure_defect", closure),
        ("operator_action_closure_norm", residual),
    ] {
        values.insert(name.into(), scale(&value, e)?);
    }
    let eigen = I::from_float(&scalar(&s.eigenvalue, s.precision)?, p)?;
    values.insert("retained_weil_energy".into(), eigen.clone());
    let mut zero_trial = false;
    if let Some(trial) = trial {
        if let Some(q) = vector(trial, p)? {
            let aq = dense_action(&m.entries, &q, e, p)?;
            values.insert(
                "finite_projected_trial_energy".into(),
                scale(&dot(&q, &aq, p).div(&norm2(&q, p))?, e)?,
            );
        } else {
            zero_trial = true;
        }
    }
    if let Some(d) = &input.deficit {
        let d = I::from_float(&scalar(d, input.precision_bits)?, p)?;
        values.insert("reference_deficit".into(), d.clone());
        if d.is_strictly_positive() {
            values.insert("signed_weil_over_deficit".into(), eigen.div(&d)?);
            if let Some(trial) = values.get("finite_projected_trial_energy").cloned() {
                values.insert("signed_trial_over_deficit".into(), trial.div(&d)?);
            }
        }
    }
    if !eigen.contains_zero() {
        if let Some(trial) = values.get("finite_projected_trial_energy").cloned() {
            values.insert("signed_trial_over_weil".into(), trial.div(&eigen)?);
        }
    }
    let energies = energies
        .iter()
        .map(|x| scale(x, e))
        .collect::<Result<Vec<_>>>()?;
    for value in values.values().chain(&energies) {
        value.validate()?;
        let magnitude = value
            .lower()
            .clone()
            .abs()
            .max(&value.upper().clone().abs());
        // Relative width for resolved nonzero values; absolute width around zero.
        let natural = if value.contains_zero() {
            Float::with_val(p, 1)
        } else {
            magnitude
        };
        if !narrow(std::slice::from_ref(value), &natural, requested)? {
            return Ok(None);
        }
    }
    Ok(Some(Measurement {
        values,
        energies,
        arithmetic_precision: p,
        zero_trial,
    }))
}

pub(super) fn measure(
    s: &RetainedState,
    m: &RetainedMatrix<'_>,
    input: &ExternalResearchInputs,
    requested: u32,
) -> Result<Option<Measurement>> {
    if requested < s.precision.max(m.precision).max(input.precision_bits) {
        bail!("energy source precision reduction")
    }
    let operators = input
        .components
        .iter()
        .map(|c| Operator::parse(c, input.precision_bits))
        .collect::<Result<Vec<_>>>()?;
    let compact = if operators.is_empty() {
        input
            .run_once
            .as_ref()
            .map(|r| {
                r.component_actions
                    .iter()
                    .map(|c| coeffs(&c.action, input.precision_bits))
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    let trial = input
        .target
        .as_ref()
        .and_then(|t| t.trial_coefficients.as_ref())
        .map(|q| coeffs(q, input.precision_bits))
        .transpose()?;
    let e = m
        .entries
        .iter()
        .filter_map(Float::get_exp)
        .map(i64::from)
        .chain(operators.iter().map(Operator::exponent))
        .chain(
            compact
                .iter()
                .flatten()
                .filter_map(Float::get_exp)
                .map(i64::from),
        )
        .max()
        .unwrap_or(0);
    for guard in [64, 128, 256, 512, 1024, 2048, 4096] {
        if let Some(result) = calculate(
            s,
            m,
            input,
            &operators,
            &compact,
            trial.as_deref(),
            e,
            requested,
            requested + guard,
        )? {
            return Ok(Some(result));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod energy_workspace_tests {
    use super::*;
    use serde_json::json;
    use xc_cache::ContentDigest;

    fn input() -> ExternalResearchInputs {
        serde_json::from_value(json!({
            "schema_version":1,"source_eigenpair":ContentDigest::sha256(b"planner-state"),
            "lambda_squared":"9","n_modes":1,"precision_bits":128,
            "convention_id":"planner-only shape","definition_digest":ContentDigest::sha256(b"planner-input"),
            "approximation_scope":"no arithmetic performed"
        })).unwrap()
    }

    fn shape(input: &ExternalResearchInputs) -> WorkspaceShape {
        WorkspaceShape::from_input(3, "3", "9", input).unwrap()
    }

    #[test]
    fn campaign_dimensions_charge_vectors_without_allocating_tau() {
        for (modes, precision) in [(500u64, 3386u32), (800, 3386), (970, 6708)] {
            let n = 2 * modes + 1;
            let shape = WorkspaceShape {
                dimension: n,
                parsed_scalars: 4 * n, // Three compact actions and a trial.
                parsed_vectors: 4,
                components: 3,
                largest_literal: 2048,
                report_text: 4096,
                ..WorkspaceShape::default()
            };
            let bytes = shape.bytes(precision, precision + 64).unwrap();
            assert!(bytes < 256 << 20, "N={modes}: {bytes}");
            // The former estimate charged 64 dense copies of borrowed Tau.
            let old = u128::from(n * n + 256 * n)
                * 64
                * (u128::from(precision + 64 + 4096).div_ceil(8) + 128);
            assert!(old > 64u128 << 30);
            let doubled = WorkspaceShape {
                dimension: 2 * n,
                parsed_scalars: 8 * n,
                ..shape
            }
            .bytes(precision, precision + 64)
            .unwrap();
            assert!(doubled > bytes && doubled < 2 * bytes);
        }
    }

    #[test]
    fn owned_compact_dense_rank_and_trial_shapes_are_counted_before_parsing() {
        let mut input = input();
        input.run_once = Some(serde_json::from_value(json!({"component_actions":[
            {"label":"a","source_digest":ContentDigest::sha256(b"a"),"action":["0","1","0"],"convention":"first"},
            {"label":"b","source_digest":ContentDigest::sha256(b"b"),"action":["0","2","0"],"convention":"second"}
        ]})).unwrap());
        let compact = shape(&input);
        assert_eq!(
            (
                compact.parsed_scalars,
                compact.parsed_vectors,
                compact.components
            ),
            (6, 2, 2)
        );
        input.components.push(OperatorComponent {
            label: "dense and rank".into(),
            source_digest: ContentDigest::sha256(b"operator"),
            diagonal: vec!["1".into(); 3],
            dense: vec!["0".into(); 9],
            rank_one: vec![super::super::RankOneTerm {
                weight: "2".into(),
                vector: vec!["1".into(); 3],
            }],
        });
        let owned = shape(&input);
        // Compact actions are not parsed when explicit operators are selected.
        assert_eq!(
            (
                owned.parsed_scalars,
                owned.parsed_vectors,
                owned.components,
                owned.rank_terms
            ),
            (16, 3, 1, 1)
        );
        input.target = Some(
            serde_json::from_value(json!({
                "definition_digest":ContentDigest::sha256(b"trial"),"evaluation_policy":"finite",
                "approximation_scope":"planner only","intervals":8,"values":vec!["1";9],
                "basis_values":[],"fixed_second_component":null,"raw_normalizer":"1",
                "trial_coefficients":["0","not parsed by the planner","0"]
            }))
            .unwrap(),
        );
        let trial = shape(&input);
        assert_eq!((trial.parsed_scalars, trial.parsed_vectors), (19, 4));
        assert_eq!(
            trial.largest_literal,
            "not parsed by the planner".len() as u64
        );
        assert!(trial.bytes(128, 192).unwrap() > owned.bytes(128, 192).unwrap());
    }

    #[test]
    fn planner_checks_overflow_and_charges_precision_literals_and_output() {
        let base = WorkspaceShape {
            dimension: 3,
            components: 1,
            ..WorkspaceShape::default()
        };
        let bytes = base.bytes(128, 192).unwrap();
        assert!(base.bytes(128, 4096).unwrap() > bytes);
        assert!(
            WorkspaceShape {
                largest_literal: 1 << 20,
                ..base
            }
            .bytes(128, 192)
            .unwrap()
                > bytes + (63 << 20)
        );
        assert!(
            WorkspaceShape {
                report_text: 1 << 20,
                ..base
            }
            .bytes(128, 192)
            .unwrap()
                > bytes + (47 << 20)
        );
        assert!(WorkspaceShape {
            parsed_scalars: u64::MAX,
            ..base
        }
        .bytes(128, 192)
        .is_err());
        assert!(WorkspaceShape {
            components: u64::MAX,
            ..base
        }
        .bytes(128, 192)
        .is_err());
        assert!(base.bytes(128, u32::MAX).is_err());
        let mut total = u64::MAX;
        assert!(add_count(&mut total, 1).is_err());
    }
}
