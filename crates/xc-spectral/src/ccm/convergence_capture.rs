//! Point diagnostics for complete retained-run research capture.
//! These producers do not change primary solves or establish infinite-limit claims.
use super::extended_research::{
    matvec, missing, put, report, row, source_unit, unresolved, ExtendedAnalysis, ExtensionOptions,
    ExternalResearchInputs,
};
use super::retained_evidence::{
    dot, norm2, scalar, transform_terms, RetainedMatrix, RetainedRoots,
};
use super::state_geometry::RetainedState;
use anyhow::{bail, Result};
use rayon::prelude::*;
use rug::{float::Constant, ops::Pow, Assign, Float};
use serde::{Deserialize, Serialize};
use xc_cache::ContentDigest;

pub const DIAGNOSTICS: &[&str] = &[
    "complex_transform",
    "root_transport",
    "operator_cluster",
    "finite_section_transfer",
    "tail_operator",
    "observable_budget",
];

/// Compact matrix actions. Values use the signed, center-oriented unit coefficient state.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct OperatorAction {
    pub label: String,
    pub source_digest: ContentDigest,
    pub action: Vec<String>,
    pub convention: String,
}
/// A fixed-basis numerical model; neither its energy nor its band is a solve input.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TailForm {
    /// When present, columns are exactly 1,z,... in this declared coordinate,
    /// after any fixed prefix already included in the forms. Enables band roots.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub polynomial_coordinate: Option<String>,
    pub definition_digest: ContentDigest,
    pub dimension: usize,
    pub finite_zero_form: Vec<String>,
    pub tail_correction: Vec<String>,
    pub lattice_gram: Vec<String>,
    pub tail_operator_error: Option<String>,
    pub coverage: String,
    pub hypotheses: Vec<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ComparisonState {
    pub source_digest: ContentDigest,
    pub matrix_digest: ContentDigest,
    pub lambda_squared: String,
    pub n_modes: usize,
    pub precision_bits: u32,
    pub coefficients: Vec<String>,
    pub matrix: Vec<String>,
    pub eigenvalue: String,
    pub assembly_policy: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SourceUncertainty {
    pub unit_state_l2_error: String,
    pub source_certificate_digest: ContentDigest,
    pub hypotheses: Vec<String>,
}
/// Optional numerical inputs bound to the parent external-source artifact.
/// The Toolkit never executes an external target formula.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RunOnceInputs {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion: Option<super::research_completion::CompletionInputs>,
    #[serde(default)]
    pub component_actions: Vec<OperatorAction>,
    #[serde(default)]
    pub derivative_actions: Vec<OperatorAction>,
    /// d(log C)/ds for the derivative actions, in the same Fourier coordinates.
    pub log_cutoff_velocity: Option<String>,
    #[serde(default)]
    pub reference_vectors: Vec<Vec<String>>,
    pub comparison: Option<ComparisonState>,
    pub tail_form: Option<TailForm>,
    pub uncertainty: Option<SourceUncertainty>,
    #[serde(default)]
    pub producer_notes: Vec<String>,
}
impl RunOnceInputs {
    pub fn validate(&self, dim: usize, p: u32) -> Result<()> {
        if let Some(c) = &self.completion {
            c.validate(dim, p)?;
        }
        if self.component_actions.len() > 64
            || self.derivative_actions.len() > 64
            || self.reference_vectors.len() > 32
            || self.producer_notes.len() > 64
        {
            bail!("run-once input count exceeded");
        }
        let check = |v: &[String]| -> Result<()> {
            for s in v {
                scalar(s, p)?;
            }
            Ok(())
        };
        for list in [&self.component_actions, &self.derivative_actions] {
            let mut labels = std::collections::BTreeSet::new();
            for a in list {
                if a.action.len() != dim
                    || !a.source_digest.validate()
                    || a.label.is_empty()
                    || a.label.len() > 128
                    || a.convention.is_empty()
                    || !labels.insert(&a.label)
                {
                    bail!("invalid compact operator action");
                }
                check(&a.action)?;
            }
        }
        if let Some(x) = &self.log_cutoff_velocity {
            scalar(x, p)?;
        }
        for v in &self.reference_vectors {
            if v.len() != dim {
                bail!("reference vector dimension mismatch");
            }
            check(v)?;
        }
        if let Some(c) = &self.comparison {
            let n = c
                .n_modes
                .checked_mul(2)
                .and_then(|n| n.checked_add(1))
                .ok_or_else(|| anyhow::anyhow!("comparison dimension overflow"))?;
            if n > dim
                || c.matrix.len() != n * n
                || c.coefficients.len() != n
                || c.matrix.len() > 4_000_000
                || !c.source_digest.validate()
                || !c.matrix_digest.validate()
                || c.assembly_policy.is_empty()
                || !(64..=p).contains(&c.precision_bits)
            {
                bail!("invalid comparison source");
            }
            check(&c.coefficients)?;
            check(&c.matrix)?;
            scalar(&c.eigenvalue, p)?;
            scalar(&c.lambda_squared, p)?;
        }
        if let Some(f) = &self.tail_form {
            let n = f.dimension;
            if f.polynomial_coordinate
                .as_ref()
                .is_some_and(|v| v.is_empty())
            {
                bail!("tail polynomial coordinate is empty");
            }
            if n == 0
                || n > 128
                || f.finite_zero_form.len() != n * n
                || f.tail_correction.len() != n * n
                || f.lattice_gram.len() != n * n
                || !f.definition_digest.validate()
                || f.coverage.is_empty()
            {
                bail!("invalid tail bilinear model");
            }
            for a in [&f.finite_zero_form, &f.tail_correction, &f.lattice_gram] {
                check(a)?;
                for i in 0..n {
                    for j in 0..i {
                        if scalar(&a[i * n + j], p)? != scalar(&a[j * n + i], p)? {
                            bail!("tail matrix is not symmetric");
                        }
                    }
                }
            }
            if let Some(e) = &f.tail_operator_error {
                if scalar(e, p)? < 0 {
                    bail!("negative tail error");
                }
            }
        }
        if let Some(u) = &self.uncertainty {
            if !u.source_certificate_digest.validate()
                || scalar(&u.unit_state_l2_error, p)? < 0
                || u.hypotheses.is_empty()
            {
                bail!("invalid declared state uncertainty");
            }
        }
        let scalars = self
            .component_actions
            .iter()
            .chain(&self.derivative_actions)
            .map(|x| x.action.len())
            .sum::<usize>()
            + self.reference_vectors.iter().map(Vec::len).sum::<usize>()
            + self
                .comparison
                .as_ref()
                .map_or(0, |c| c.matrix.len() + c.coefficients.len())
            + self
                .tail_form
                .as_ref()
                .map_or(0, |f| 3 * f.dimension * f.dimension);
        if scalars > 4_000_000 {
            bail!("run-once scalar budget exceeded");
        }
        Ok(())
    }
}
fn vals(v: &[String], p: u32) -> Result<Vec<Float>> {
    v.iter().map(|s| scalar(s, p)).collect()
}
fn input(i: Option<&ExternalResearchInputs>) -> Option<&RunOnceInputs> {
    i.and_then(|x| x.run_once.as_ref())
}

// Minimal complex arithmetic retains MPFR precision without enabling a new scalar backend.
#[derive(Clone)]
struct Z {
    re: Float,
    im: Float,
}
impl Z {
    fn new(p: u32, re: i32, im: i32) -> Self {
        Self {
            re: Float::with_val(p, re),
            im: Float::with_val(p, im),
        }
    }
    fn add(&self, b: &Self, p: u32) -> Self {
        Self {
            re: Float::with_val(p, &self.re) + &b.re,
            im: Float::with_val(p, &self.im) + &b.im,
        }
    }
    fn sub(&self, b: &Self, p: u32) -> Self {
        Self {
            re: Float::with_val(p, &self.re) - &b.re,
            im: Float::with_val(p, &self.im) - &b.im,
        }
    }
    fn mul(&self, b: &Self, p: u32) -> Self {
        Self {
            re: Float::with_val(p, &self.re) * &b.re - Float::with_val(p, &self.im) * &b.im,
            im: Float::with_val(p, &self.re) * &b.im + Float::with_val(p, &self.im) * &b.re,
        }
    }
    fn scale(&self, b: &Float, p: u32) -> Self {
        Self {
            re: Float::with_val(p, &self.re) * b,
            im: Float::with_val(p, &self.im) * b,
        }
    }
    fn div(&self, b: &Self, p: u32) -> Self {
        let den = Float::with_val(p, &b.re).square() + Float::with_val(p, &b.im).square();
        Self {
            re: (Float::with_val(p, &self.re) * &b.re + Float::with_val(p, &self.im) * &b.im)
                / &den,
            im: (Float::with_val(p, &self.im) * &b.re - Float::with_val(p, &self.re) * &b.im) / den,
        }
    }
    fn abs(&self, p: u32) -> Float {
        (Float::with_val(p, &self.re).square() + Float::with_val(p, &self.im).square()).sqrt()
    }
    fn sincos(&self, p: u32) -> (Self, Self) {
        let (a, b) = self.re.clone().sin_cos(Float::new(p));
        let h = self.im.clone().sinh();
        let c = self.im.clone().cosh();
        (
            Self {
                re: Float::with_val(p, &a) * &c,
                im: Float::with_val(p, &b) * &h,
            },
            Self {
                re: b * c,
                im: -a * h,
            },
        )
    }
}
fn sinc(z: &Z, p: u32) -> (Z, Z) {
    if z.abs(p) < Float::with_val(p, 0.5) {
        let z2 = z.mul(z, p);
        let mut term = Z::new(p, 1, 0);
        let mut sum = term.clone();
        let mut der = Z::new(p, 0, 0);
        if z.abs(p) == 0 {
            return (sum, der);
        }
        for n in 1..=p {
            let f = -Float::with_val(p, 1) / (2 * n) / (2 * n + 1);
            term = term.mul(&z2, p).scale(&f, p);
            sum = sum.add(&term, p);
            der = der.add(&term.div(z, p).scale(&Float::with_val(p, 2 * n), p), p);
            if term.abs(p) < (Float::with_val(p, 1) >> p) {
                break;
            }
        }
        (sum, der)
    } else {
        let (s, c) = z.sincos(p);
        let a = s.div(z, p);
        let d = c.sub(&a, p).div(z, p);
        (a, d)
    }
}
fn complex_value(s: &RetainedState, z: &Z, p: u32) -> Result<(Z, Z, Float)> {
    let l = scalar(&s.cutoff, p)?.ln();
    let half = Float::with_val(p, &l) / 2;
    let v = source_unit(s, p);
    let pi = Float::with_val(p, Constant::Pi);
    let mut f = Z::new(p, 0, 0);
    let mut d = f.clone();
    let mut abs = Float::with_val(p, 0);
    for (idx, c) in v.iter().enumerate() {
        let j = idx as i64 - s.modes as i64;
        let mut q = z.scale(&half, p);
        q.re += Float::with_val(p, &pi) * j;
        let (a, b) = sinc(&q, p);
        let sign = if j.unsigned_abs().is_multiple_of(2) {
            1
        } else {
            -1
        };
        let factor = Float::with_val(p, c) * l.clone().sqrt() * sign;
        let a = a.scale(&factor, p);
        f = f.add(&a, p);
        abs += a.abs(p);
        d = d.add(&b.scale(&(factor * &half), p), p);
    }
    Ok((f, d, abs))
}
fn complex(
    s: &RetainedState,
    roots: Option<&RetainedRoots>,
    o: &ExtensionOptions,
) -> Result<ExtendedAnalysis> {
    let p = o.working_precision_bits;
    let mut out = report("complex_transform", s, o);
    let mut points = vec![(0, Float::with_val(p, 0))];
    if let Some(r) = roots {
        for a in &r.dataset.points {
            if let Some(v) = &a.value {
                points.push((a.ordinal, scalar(v, p)?));
            }
        }
    }
    let (anchor, _, _) = complex_value(s, &Z::new(p, 0, 0), p)?;
    put(&mut out.values, "normalization_anchor", &anchor.re);
    out.convention="unit_L2_dx and F(z)/F(0); exp(i*z*x); real retained coefficients; offsets 0,+/-0.25,+/-1 at every available root and origin; closed 16-interval-per-side rectangle; samples are not contour certificates".into();
    let mut grid = points
        .iter()
        .flat_map(|(ordinal, t)| {
            [-4, -1, 0, 1, 4]
                .into_iter()
                .map(move |j| (*ordinal, t.clone(), j))
        })
        .collect::<Vec<_>>();
    let right = points
        .iter()
        .map(|(_, t)| t.clone())
        .max_by(|a, b| a.total_cmp(b))
        .unwrap()
        + 1u32;
    let left = Float::with_val(p, -1);
    let width = Float::with_val(p, &right) - &left;
    let probe_count = grid.len();
    // Sixteen intervals per side, closed and counterclockwise. Derivative values
    // support later adaptive checking; sampling alone never certifies a zero count.
    let mut contour = Vec::new();
    for side in 0..4 {
        for j in 0..16 {
            let a = Float::with_val(p, j) / 16u32;
            let (re, im) = match side {
                0 => (
                    Float::with_val(p, &left) + &width * &a,
                    Float::with_val(p, -1),
                ),
                1 => (
                    right.clone(),
                    Float::with_val(p, -1) + Float::with_val(p, &a) * 2u32,
                ),
                2 => (
                    Float::with_val(p, &right) - &width * &a,
                    Float::with_val(p, 1),
                ),
                _ => (
                    left.clone(),
                    Float::with_val(p, 1) - Float::with_val(p, &a) * 2u32,
                ),
            };
            contour.push((re, im));
        }
    }
    contour.push(contour[0].clone());
    grid.extend(contour.iter().map(|(re, _)| (0, re.clone(), 0)));
    let checkpoints = super::capture_runtime::Checkpoints::new(&(
        "complex-rows-v2",
        &s.manifest.content_digest,
        roots.map(|r| &r.manifest.content_digest),
        o,
    ))?;
    out.rows =
        super::capture_runtime::row_blocks(&checkpoints, grid.len(), |index| -> Result<_> {
            let (ordinal, t, j) = grid[index].clone();
            let z = if index < probe_count {
                Z {
                    re: t,
                    im: Float::with_val(p, j) / 4,
                }
            } else {
                let (re, im) = &contour[index - probe_count];
                Z {
                    re: re.clone(),
                    im: im.clone(),
                }
            };
            let (f, d, a) = complex_value(s, &z, p)?;
            let mut row = row(
                index + 1,
                if index < probe_count {
                    "complex_transform_sample"
                } else {
                    "contour_sample"
                },
            );
            for (name, value) in [
                ("input_ordinal", Float::with_val(p, ordinal)),
                ("z_re", z.re),
                ("z_im", z.im),
                ("value_re", f.re.clone()),
                ("value_im", f.im.clone()),
                ("derivative_re", d.re.clone()),
                ("derivative_im", d.im.clone()),
                ("sum_absolute_terms", a.clone()),
            ] {
                put(&mut row.values, name, &value);
            }
            let guard = (a + 1u32) >> (p - 32);
            if anchor.abs(p) > guard {
                let q = f.div(&anchor, p);
                put(&mut row.values, "normalized_re", &q.re);
                put(&mut row.values, "normalized_im", &q.im);
            } else {
                row.outcome = "unresolved_denominator".into();
            }
            if f.abs(p) > guard {
                let q = d.div(&f, p);
                put(&mut row.values, "log_derivative_re", &q.re);
                put(&mut row.values, "log_derivative_im", &q.im);
            }
            Ok(row)
        })?;
    Ok(out)
}

fn transfer(
    s: &RetainedState,
    m: Option<&RetainedMatrix<'_>>,
    o: &ExtensionOptions,
    i: Option<&RunOnceInputs>,
) -> Result<ExtendedAnalysis> {
    let mut out = report("finite_section_transfer", s, o);
    let Some(m) = m else {
        return Ok(missing(out, "retained Tau required"));
    };
    let p = o.working_precision_bits;
    let v = source_unit(s, p);
    let n = v.len();
    let e = scalar(&s.eigenvalue, p)?;
    let mut action = vec![Float::with_val(p, 0); n];
    let mut mass = Float::with_val(p, 0);
    // Expanding symmetric Fourier prefixes cost O(n^2), not a new solve per prefix.
    for k in 0..=s.modes {
        let columns = if k == 0 {
            vec![s.modes]
        } else {
            vec![s.modes - k, s.modes + k]
        };
        for c in columns {
            mass += Float::with_val(p, &v[c]).square();
            for (row, x) in action.iter_mut().enumerate() {
                *x += Float::with_val(p, &m.entries[row * n + c]) * &v[c];
            }
        }
        let lo = s.modes - k;
        let hi = s.modes + k;
        let mut low = Float::with_val(p, 0);
        let mut high = Float::with_val(p, 0);
        let mut energy = Float::with_val(p, 0);
        for r in 0..n {
            if (lo..=hi).contains(&r) {
                low += (Float::with_val(p, &action[r]) - Float::with_val(p, &e) * &v[r]).square();
                energy += Float::with_val(p, &v[r]) * &action[r];
            } else {
                high += Float::with_val(p, &action[r]).square();
            }
        }
        let mut rr = row(k + 1, "projection_of_retained_full_state");
        for (name, value) in [
            ("n_modes", Float::with_val(p, k)),
            ("retained_mass", mass.clone()),
            ("omitted_mass", Float::with_val(p, 1) - &mass),
            ("low_residual_squared", low),
            ("high_forcing_squared", high),
            ("truncated_energy", energy),
        ] {
            put(&mut rr.values, name, &value);
        }
        out.rows.push(rr);
    }
    if let Some(c) = i.and_then(|x| x.comparison.as_ref()) {
        if scalar(&c.lambda_squared, p)? != scalar(&s.cutoff, p)? {
            bail!("comparison cutoff differs");
        }
        let d = 2 * c.n_modes + 1;
        let offset = s.modes - c.n_modes;
        let small = vals(&c.matrix, p)?;
        let cv = vals(&c.coefficients, p)?;
        let norm = norm2(&cv, p).sqrt();
        if norm == 0 {
            bail!("zero comparison state");
        }
        let cv = cv.into_iter().map(|x| x / &norm).collect::<Vec<_>>();
        let mut delta2 = Float::with_val(p, 0);
        let mut signed = Float::with_val(p, 0);
        for a in 0..d {
            for b in 0..d {
                let delta = Float::with_val(p, &m.entries[(a + offset) * n + b + offset])
                    - &small[a * d + b];
                delta2 += delta.clone().square();
                signed += delta * &cv[a] * &cv[b];
            }
        }
        put(
            &mut out.values,
            "comparison_block_frobenius_difference",
            &delta2.sqrt(),
        );
        put(
            &mut out.values,
            "comparison_state_signed_block_defect",
            &signed,
        );
        put(
            &mut out.values,
            "comparison_precision_bits",
            &Float::with_val(p, c.precision_bits),
        );
    }
    out.convention="all symmetric prefixes of one retained Tau and projections of its state; no independent prefix eigenstate claim; optional separately supplied comparison block".into();
    if i.and_then(|i| i.comparison.as_ref()).is_none() {
        out.reason=Some("independently assembled comparison configuration unavailable; every parent-derived prefix retained".into());
    }
    Ok(out)
}

fn budget(
    s: &RetainedState,
    roots: Option<&RetainedRoots>,
    o: &ExtensionOptions,
    i: Option<&RunOnceInputs>,
) -> Result<ExtendedAnalysis> {
    let mut out = report("observable_budget", s, o);
    let p = o.working_precision_bits;
    let l = scalar(&s.cutoff, p)?.ln();
    let (a, _, aa, _) = transform_terms(s, &Float::with_val(p, 0), p)?;
    put(&mut out.values, "transform_origin", &a);
    put(&mut out.values, "origin_absolute_terms", &aa);
    let uncertainty = i.and_then(|x| x.uncertainty.as_ref());
    let error = uncertainty
        .map(|u| scalar(&u.unit_state_l2_error, p))
        .transpose()?;
    if let Some(e) = &error {
        put(
            &mut out.values,
            "declared_origin_error",
            &(Float::with_val(p, e) * l.clone().sqrt()),
        );
        put(
            &mut out.values,
            "conditional_origin_lower_margin",
            &(a.clone().abs() - e * l.clone().sqrt()),
        );
    }
    if let Some(r) = roots {
        for point in &r.dataset.points {
            let mut rr = row(point.ordinal, "retained_window_ordinal");
            let Some(t) = &point.value else {
                rr.outcome = "missing_input".into();
                out.rows.push(rr);
                continue;
            };
            let t = scalar(t, p)?;
            let (f, d, abs, absd) = transform_terms(s, &t, p)?;
            for (name, x) in [
                ("t", t),
                ("value", f.clone()),
                ("derivative", d.clone()),
                ("absolute_value_terms", abs),
                ("absolute_derivative_terms", absd),
            ] {
                put(&mut rr.values, name, &x);
            }
            if let Some(e) = &error {
                let ve = e * l.clone().sqrt();
                let de = e * (Float::with_val(p, &l).pow(3u32) / 12u32).sqrt();
                put(&mut rr.values, "conditional_value_error", &ve);
                put(&mut rr.values, "conditional_derivative_error", &de);
                put(
                    &mut rr.values,
                    "conditional_slope_lower_margin",
                    &(d.abs() - de),
                );
                rr.notes.push("declared source certificate and hypotheses retained; isolation and ordinal certificate not inferred".into());
            } else {
                rr.outcome = "channels_resolved_budget_unassessed".into();
                rr.notes.push("source error unavailable; additional arithmetic precision is not source accuracy".into());
            }
            out.rows.push(rr);
        }
    }
    out.convention="unit_L2_dx source uncertainty; Cauchy-Schwarz value and derivative transport; retained-window ordinals remain distinct from zeta ordinals; conditional expressions only".into();
    Ok(out)
}

#[allow(clippy::too_many_arguments)] // Independently admitted sources and reused directional evidence.
pub(super) fn analyze(
    id: &str,
    s: &RetainedState,
    m: Option<&RetainedMatrix<'_>>,
    roots: Option<&RetainedRoots>,
    o: &ExtensionOptions,
    i: Option<&ExternalResearchInputs>,
    directional: Option<&ExtendedAnalysis>,
) -> Result<ExtendedAnalysis> {
    match id {
        "complex_transform" => complex(s, roots, o),
        "finite_section_transfer" => transfer(s, m, o, input(i)),
        "observable_budget" => budget(s, roots, o, input(i)),
        "operator_cluster" => cluster_operator(s, m, o, input(i)),
        "tail_operator" => super::atom_research::tail_report(s, o, i),
        "root_transport" => transport(s, o, input(i), directional),
        _ => bail!("unknown convergence diagnostic"),
    }
}
fn transport(
    s: &RetainedState,
    o: &ExtensionOptions,
    i: Option<&RunOnceInputs>,
    directional: Option<&ExtendedAnalysis>,
) -> Result<ExtendedAnalysis> {
    let mut out = report("root_transport", s, o);
    let Some(directional) = directional else {
        return Ok(missing(out, "retained Tau and roots required"));
    };
    out.rows = directional.rows.clone();
    out.outcome = directional.outcome.clone();
    out.reason = directional.reason.clone();
    let p = o.working_precision_bits;
    let l = scalar(&s.cutoff, p)?.ln();
    let pi = Float::with_val(p, Constant::Pi);
    let actions = i.map(|i| i.derivative_actions.as_slice()).unwrap_or(&[]);
    let dl = i
        .and_then(|x| x.log_cutoff_velocity.as_ref())
        .map(|x| scalar(x, p))
        .transpose()?;
    for rr in &mut out.rows {
        let mut sum = Float::with_val(p, 0);
        let mut abs = Float::with_val(p, 0);
        for action in actions {
            if action.label == "tau_total" {
                continue;
            }
            if let Some(f) = rr.values.get(&format!("forcing_{}", action.label)) {
                let f = scalar(f, p)?;
                sum += &f;
                abs += f.abs();
            }
        }
        put(&mut rr.values, "component_forcing_sum", &sum);
        put(&mut rr.values, "absolute_component_forcing_sum", &abs);
        if let Some(f) = rr.values.get("forcing_tau_total") {
            let f = scalar(f, p)?;
            put(&mut rr.values, "forcing_closure_defect", &(f - sum));
        }
        if let (Some(response), Some(t), Some(dl)) = (
            rr.values.get("conditional_tau_response_tau_total"),
            rr.values.get("t"),
            dl.as_ref(),
        ) {
            let response = scalar(response, p)?;
            let t = scalar(t, p)?;
            if t != 0 {
                let geometry = -Float::with_val(p, &t) * dl / &l;
                let motion = response * 2u32 * pi.clone().square() / (&t * l.clone().square());
                put(&mut rr.values, "support_motion", &geometry);
                put(&mut rr.values, "operator_motion", &motion);
                put(
                    &mut rr.values,
                    "conditional_total_physical_velocity",
                    &(geometry + motion),
                );
            }
        }
    }
    if let Some(c) = i.and_then(|i| i.completion.as_ref()) {
        for rr in &mut out.rows {
            if let Some(check) = c.response_checks.iter().find(|c| c.ordinal == rr.ordinal) {
                let matches=check.branch=="production_shifted_secular" && check.coordinate=="mellin_t" && check.derivative_parameter=="u=log(lambda_squared)" && check.activation_convention=="analytic_right_continuous_active_prime_set; tau=pole-archimedean-prime; total roots include d(2*pi*n/u)/du=-2*pi*n/u^2" && rr.values.get("t").is_some_and(|t|scalar(t,p).ok()==scalar(&check.t,p).ok());
                if matches {
                    if let Some(v) = &check.support_velocity {
                        put(
                            &mut rr.values,
                            "retained_secular_pole_motion",
                            &scalar(v, p)?,
                        );
                    }
                    if let Some(v) = &check.total_velocity {
                        put(&mut rr.values, "retained_total_velocity", &scalar(v, p)?);
                    }
                    if let (Some(a), Some(b), Some(total)) = (
                        &check.fixed_velocity,
                        &check.support_velocity,
                        &check.total_velocity,
                    ) {
                        put(
                            &mut rr.values,
                            "retained_transport_additivity_defect",
                            &(scalar(a, p)? + scalar(b, p)? - scalar(total, p)?),
                        );
                    }
                    // The divided-state formula assumes an unshifted rational root.
                    // Do not silently equate a shifted secular derivative with it.
                    rr.notes.push(format!("retained shifted-secular response source {}; conditional unshifted directional formula is a distinct branch; cross-formula equality not asserted",check.source_digest.0));
                } else {
                    rr.notes.push("retained response comparison withheld: coordinate, parameter, activation convention or exact evaluation point differs".into());
                }
            }
        }
    }
    out.convention="fixed Fourier dimension; d(log C)/ds declared; signed support-plus-operator response at every retained root; directional source reused; conditional simple-minimum/displacement/root hypotheses; prime-entry scope inherited from source".into();
    Ok(out)
}

fn cluster_operator(
    s: &RetainedState,
    m: Option<&RetainedMatrix<'_>>,
    o: &ExtensionOptions,
    i: Option<&RunOnceInputs>,
) -> Result<ExtendedAnalysis> {
    let mut out = report("operator_cluster", s, o);
    let Some(m) = m else {
        return Ok(missing(out, "retained Tau required"));
    };
    let p = o.working_precision_bits;
    let n = s.coefficients.len();
    let v = source_unit(s, p);
    let supplied = i.filter(|x| !x.reference_vectors.is_empty());
    let raw = if let Some(i) = supplied {
        i.reference_vectors
            .iter()
            .map(|x| vals(x, p))
            .collect::<Result<Vec<_>>>()?
    } else {
        // A reproducible finite subspace, not a claim about the prolate reference.
        let mut a = vec![v.clone()];
        for k in 0..s.modes.min(3) + 1 {
            let mut q = vec![Float::with_val(p, 0); n];
            q[s.modes + k] = Float::with_val(p, 1);
            q[s.modes - k] = Float::with_val(p, 1);
            a.push(q);
        }
        a
    };
    let mut q: Vec<Vec<Float>> = vec![];
    let threshold = Float::with_val(p, 1) >> (p / 2);
    for mut a in raw {
        for _ in 0..2 {
            for b in &q {
                let c = dot(&a, b, p);
                for (x, y) in a.iter_mut().zip(b) {
                    *x -= Float::with_val(p, &c) * y;
                }
            }
        }
        let norm = norm2(&a, p).sqrt();
        if norm > threshold {
            q.push(a.into_iter().map(|x| x / &norm).collect());
        }
    }
    if q.is_empty() {
        return Ok(unresolved(out, "reference span has no resolved vector"));
    }
    let b = q.len();
    let aq = q.iter().map(|q| matvec(m, q, p)).collect::<Vec<_>>();
    let mut small = vec![Float::with_val(p, 0); b * b];
    for a in 0..b {
        for c in 0..b {
            small[a * b + c] = dot(&q[a], &aq[c], p);
        }
    }
    let mut k = aq.clone();
    for c in 0..b {
        for a in 0..b {
            for (j, x) in k[c].iter_mut().enumerate() {
                *x -= Float::with_val(p, &q[a][j]) * &small[a * b + c];
            }
        }
    }
    let e = scalar(&s.eigenvalue, p)?;
    put(
        &mut out.values,
        "subspace_dimension",
        &Float::with_val(p, b),
    );
    put(&mut out.values, "retained_energy_shift", &e);
    let mut projected = v.clone();
    for a in &q {
        let overlap = dot(a, &v, p);
        for (x, y) in projected.iter_mut().zip(a) {
            *x -= Float::with_val(p, &overlap) * y;
        }
    }
    put(
        &mut out.values,
        "source_leakage_squared",
        &norm2(&projected, p),
    );
    // Q(A-EI)Q + UU^T; one factorization shared by every feedback column.
    let workspace = (n as u64)
        .saturating_mul(n as u64)
        .saturating_mul(3 * (u64::from(p).div_ceil(8) + 64));
    put(
        &mut out.values,
        "estimated_factorization_workspace_bytes",
        &Float::with_val(p, workspace),
    );
    let within_budget = workspace <= o.maximum_working_bytes.unwrap_or(8 * 1024 * 1024 * 1024);
    let mut h = if within_budget {
        vec![Float::with_val(p, 0); n * n]
    } else {
        Vec::new()
    };
    let mut right = aq.clone();
    for a in 0..b {
        for c in 0..n {
            right[a][c] =
                -Float::with_val(p, &aq[a][c]) + (Float::with_val(p, &e) + 1u32) * &q[a][c];
            for d in 0..b {
                right[a][c] += Float::with_val(p, &small[a * b + d]) * &q[d][c];
            }
        }
    }

    h.par_chunks_mut(n).enumerate().for_each(|(r, row)| {
        for (c, x) in row.iter_mut().enumerate() {
            *x = Float::with_val(p, &m.entries[r * n + c]);
            if r == c {
                *x -= &e;
            }
            for a in 0..b {
                *x += Float::with_val(p, &q[a][r]) * &right[a][c]
                    - Float::with_val(p, &aq[a][r]) * &q[a][c];
            }
        }
    });
    let checkpoints = super::capture_runtime::Checkpoints::new(&(
        "complement-factor-v2",
        &s.manifest.content_digest,
        &m.manifest.content_digest,
        o,
        i,
    ))?;
    let factor = if within_budget {
        let saved = checkpoints.load::<(Vec<String>, Vec<usize>)>("factor")?;
        if let Some((lu, perm)) = saved.filter(|(lu, perm)| {
            lu.len() == n * n
                && perm.len() == n
                && perm
                    .iter()
                    .copied()
                    .collect::<std::collections::BTreeSet<_>>()
                    .into_iter()
                    .eq(0..n)
        }) {
            Ok(xc_numerics::linalg::LuFactors {
                lu: vals(&lu, p)?,
                perm,
            })
        } else {
            let _stage = super::capture_runtime::Stage::new("complement factorization");
            let f = xc_numerics::linalg::lu_factor(&h, n);
            if let Ok(f) = &f {
                if let Err(e) = checkpoints.save(
                    "factor",
                    &(
                        f.lu.iter()
                            .map(xc_numerics::prefix::lossless_decimal)
                            .collect::<Vec<_>>(),
                        &f.perm,
                    ),
                ) {
                    eprintln!("factor checkpoint unavailable: {e}");
                }
            }
            f
        }
    } else {
        Err(anyhow::anyhow!("factorization workspace budget exceeded"))
    };
    let mut solves = vec![];
    let mut resolved = true;
    match factor {
        Ok(f) => {
            for (column, rhs) in k.iter().enumerate() {
                let _stage = super::capture_runtime::Stage::new(format!(
                    "complement solve/validation {column}"
                ));
                let key = format!("solve-{column}");
                let y = if let Some(v) = checkpoints
                    .load::<Vec<String>>(&key)?
                    .filter(|v| v.len() == n)
                {
                    vals(&v, p)?
                } else {
                    let y = xc_numerics::linalg::lu_solve(&f, rhs, n, p);
                    if let Err(e) = checkpoints.save(
                        &key,
                        &y.iter()
                            .map(xc_numerics::prefix::lossless_decimal)
                            .collect::<Vec<_>>(),
                    ) {
                        eprintln!("solve checkpoint unavailable: {e}");
                    }
                    y
                };
                let mut err = Float::with_val(p, 0);
                for r in 0..n {
                    err += (dot(&h[r * n..(r + 1) * n], &y, p) - &rhs[r]).square();
                }
                let relative = err.sqrt() / (norm2(rhs, p).sqrt() + 1u32);
                if !relative.is_finite() || relative > (Float::with_val(p, 1) >> (p / 3)) {
                    resolved = false;
                }
                solves.push((y, relative));
            }
        }
        Err(_) => resolved = false,
    }
    for a in 0..b {
        for c in 0..b {
            let mut rr = row(a * b + c + 1, "cluster_operator_entry");
            put(&mut rr.values, "row", &Float::with_val(p, a));
            put(&mut rr.values, "column", &Float::with_val(p, c));
            put(&mut rr.values, "compressed_operator", &small[a * b + c]);
            put(&mut rr.values, "coupling_gram", &dot(&k[a], &k[c], p));
            if resolved {
                let sigma = dot(&k[a], &solves[c].0, p);
                put(&mut rr.values, "signed_complement_feedback", &sigma);
                put(
                    &mut rr.values,
                    "effective_operator",
                    &(Float::with_val(p, &small[a * b + c]) - sigma),
                );
                put(&mut rr.values, "solve_relative_residual", &solves[c].1);
            } else {
                rr.outcome = "unresolved_denominator".into();
            }
            out.rows.push(rr);
        }
    }
    if !resolved {
        out.outcome = "partial_unresolved".into();
        out.reason=Some(if within_budget {"complement factorization or solve residual unresolved; compressed operator and coupling retained"}else{"factorization workspace budget exceeded; compressed operator and coupling retained; retry only this diagnostic with a larger explicit budget"}.into());
    }
    out.convention=if supplied.is_some(){"declared reference span; twice-reorthogonalized coordinates; signed finite complement feedback at retained E; point solves not inertia certificates"}else{"automatic span of retained state and first four even Fourier modes; not a prolate reference; signed finite complement feedback at retained E"}.into();
    Ok(out)
}

fn model_matvec(a: &[Float], v: &[Float], n: usize, p: u32) -> Vec<Float> {
    a.par_chunks(n)
        .map(|row| {
            let mut sum = Float::with_val(p, 0);
            let mut term = Float::with_val(p, 0);
            for (a, v) in row.iter().zip(v) {
                term.assign(a);
                term *= v;
                sum += &term;
            }
            sum
        })
        .collect()
}
pub(crate) fn tail_operator(
    s: &RetainedState,
    o: &ExtensionOptions,
    i: Option<&RunOnceInputs>,
) -> Result<ExtendedAnalysis> {
    let store = super::capture_runtime::Checkpoints::new(&(
        "finite-tail-model-solve-v1",
        &s.manifest.content_digest,
        o,
        i,
    ))?;
    if let Some(result) = store.load::<ExtendedAnalysis>("model-solve")? {
        return Ok(result);
    }
    let result = tail_operator_uncached(s, o, i)?;
    if result.outcome == "point_measurement" {
        if let Err(e) = store.save("model-solve", &result) {
            eprintln!("tail model checkpoint unavailable: {e}");
        }
    }
    Ok(result)
}
fn tail_operator_uncached(
    s: &RetainedState,
    o: &ExtensionOptions,
    i: Option<&RunOnceInputs>,
) -> Result<ExtendedAnalysis> {
    let mut out = report("tail_operator", s, o);
    let Some(f) = i.and_then(|i| i.tail_form.as_ref()) else {
        return Ok(missing(out,"fixed-basis zero/lattice/tail forms required; model and omitted-tail formula are not inferred from a state"));
    };
    let p = o.working_precision_bits;
    let n = f.dimension;
    let workspace = (n as u64)
        .saturating_pow(2)
        .saturating_mul(u64::from(p).div_ceil(8) + 64)
        .saturating_mul(20);
    put(
        &mut out.values,
        "estimated_model_workspace_bytes",
        &Float::with_val(p, workspace),
    );
    if workspace > o.maximum_working_bytes.unwrap_or(8 << 30) {
        return Ok(unresolved(
            out,
            "tail model, counterfactual and vector workspace exceeds diagnostic budget",
        ));
    }
    let finite = vals(&f.finite_zero_form, p)?;
    let a = finite.clone();
    let tail = vals(&f.tail_correction, p)?;
    let gram = vals(&f.lattice_gram, p)?;
    let a = a
        .into_iter()
        .zip(&tail)
        .map(|(a, b)| a + b)
        .collect::<Vec<_>>();
    let mut pivots = Vec::with_capacity(n);
    let mut chol = vec![Float::with_val(p, 0); n * n];
    for r in 0..n {
        for c in 0..=r {
            let mut v = Float::with_val(p, &gram[r * n + c]);
            for k in 0..c {
                v -= Float::with_val(p, &chol[r * n + k]) * &chol[c * n + k];
            }
            if r == c {
                if v <= 0 {
                    return Ok(unresolved(
                        out,
                        "lattice Gram Cholesky pivot is not positive",
                    ));
                }
                pivots.push(v.clone());
                chol[r * n + c] = v.sqrt();
            } else {
                chol[r * n + c] = v / &chol[c * n + c];
            }
        }
    }
    let mut inv = vec![Float::with_val(p, 0); n * n];
    for c in 0..n {
        for r in c..n {
            let mut v = Float::with_val(p, if r == c { 1 } else { 0 });
            for k in c..r {
                v -= Float::with_val(p, &chol[r * n + k]) * &inv[k * n + c];
            }
            inv[r * n + c] = v / &chol[r * n + r];
        }
    }
    let whiten = |matrix: &[Float]| -> Vec<Float> {
        let left = (0..n)
            .into_par_iter()
            .map(|r| {
                (0..n)
                    .map(|c| {
                        let mut x = Float::with_val(p, 0);
                        let mut term = Float::with_val(p, 0);
                        for k in 0..n {
                            term.assign(&inv[r * n + k]);
                            term *= &matrix[k * n + c];
                            x += &term;
                        }
                        x
                    })
                    .collect::<Vec<_>>()
            })
            .flatten()
            .collect::<Vec<_>>();
        let rows = (0..n)
            .into_par_iter()
            .map(|r| {
                (0..=r)
                    .map(|c| {
                        let mut x = Float::with_val(p, 0);
                        let mut term = Float::with_val(p, 0);
                        for k in 0..n {
                            term.assign(&left[r * n + k]);
                            term *= &inv[c * n + k];
                            x += &term;
                        }
                        x
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut result = vec![Float::with_val(p, 0); n * n];
        for r in 0..n {
            for c in 0..=r {
                result[r * n + c] = rows[r][c].clone();
                result[c * n + r] = rows[r][c].clone();
            }
        }
        result
    };
    let transformed = whiten(&a);
    let (diag, off, q) = xc_numerics::eigen::householder_tridiag_hp_stable(&transformed, n, p)?;
    let mut eigenvalues = xc_numerics::eigen::tridiag_eigenvalues_hp(&diag, &off, p)?;
    eigenvalues.sort_by(Float::total_cmp);
    let energy = Float::with_val(p, &eigenvalues[0]) * 2u32;
    put(&mut out.values, "model_energy", &energy);
    let retained = scalar(&s.eigenvalue, p)?;
    put(
        &mut out.values,
        "retained_energy_for_scoring_only",
        &retained,
    );
    let defect = Float::with_val(p, &energy) - &retained;
    put(
        &mut out.values,
        "model_energy_signed_scoring_defect",
        &defect,
    );
    if retained != 0 {
        put(
            &mut out.values,
            "model_energy_relative_scoring_defect",
            &(defect / retained.clone().abs()),
        );
    }
    let min_pivot = pivots.iter().min_by(|a, b| a.total_cmp(b)).unwrap();
    let max_pivot = pivots.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
    put(
        &mut out.values,
        "minimum_lattice_gram_cholesky_pivot",
        min_pivot,
    );
    put(
        &mut out.values,
        "maximum_lattice_gram_cholesky_pivot",
        max_pivot,
    );
    put(
        &mut out.values,
        "lattice_gram_pivot_ratio",
        &(Float::with_val(p, max_pivot) / min_pivot),
    );
    put(
        &mut out.values,
        "inverse_cholesky_frobenius_norm_squared",
        &norm2(&inv, p),
    );
    match xc_numerics::eigen::dense_symmetric_eigenvalues_hp_stable(&whiten(&finite), n, p) {
        Ok(values) => {
            let without = values.into_iter().min_by(Float::total_cmp).unwrap() * 2u32;
            put(&mut out.values, "model_energy_without_tail", &without);
            put(
                &mut out.values,
                "tail_energy_lift",
                &(Float::with_val(p, &energy) - &without),
            );
            if energy != 0 {
                put(
                    &mut out.values,
                    "relative_tail_energy_lift",
                    &((Float::with_val(p, &energy) - &without) / energy.clone().abs()),
                );
            }
        }
        Err(_) => {
            out.outcome = "partial_unresolved".into();
            out.reason = Some(
                "no-tail counterfactual solve unresolved; completed model spectrum retained".into(),
            );
        }
    }
    let options = xc_numerics::eigen::TridiagEigvecOptions {
        max_steps: 200,
        early_termination: true,
        solver: xc_numerics::eigen::TridiagSolver::BandedInterleaved,
    };
    let vector = xc_numerics::eigen::tridiag_eigenvector_for_value_hp(
        &diag,
        &off,
        &eigenvalues[0],
        p,
        options,
    );
    match vector {
        Ok(v) => {
            let y = model_matvec(&q, &v, n, p);
            let mut coeff = vec![Float::with_val(p, 0); n];
            for c in 0..n {
                for r in 0..n {
                    coeff[c] += Float::with_val(p, &inv[r * n + c]) * &y[r];
                }
            }
            let gv = model_matvec(&gram, &coeff, n, p);
            let norm = dot(&coeff, &gv, p);
            if norm > 0 {
                for x in &mut coeff {
                    *x /= norm.clone().sqrt();
                }
                let av = model_matvec(&a, &coeff, n, p);
                let gv = model_matvec(&gram, &coeff, n, p);
                let residual = av
                    .iter()
                    .zip(&gv)
                    .map(|(a, g)| Float::with_val(p, a) - Float::with_val(p, &eigenvalues[0]) * g)
                    .collect::<Vec<_>>();
                let rn = norm2(&residual, p).sqrt();
                let scale =
                    norm2(&av, p).sqrt() + eigenvalues[0].clone().abs() * norm2(&gv, p).sqrt();
                put(&mut out.values, "model_vector_absolute_residual", &rn);
                if scale > 0 {
                    put(
                        &mut out.values,
                        "model_vector_relative_residual",
                        &(Float::with_val(p, &rn) / &scale),
                    );
                }
                put(
                    &mut out.values,
                    "model_vector_lattice_norm_squared",
                    &dot(&coeff, &gv, p),
                );
                put(
                    &mut out.values,
                    "model_vector_zero_energy",
                    &(dot(&coeff, &model_matvec(&finite, &coeff, n, p), p) * 2u32),
                );
                put(
                    &mut out.values,
                    "model_vector_tail_energy",
                    &(dot(&coeff, &model_matvec(&tail, &coeff, n, p), p) * 2u32),
                );
                for (index, x) in coeff.iter().enumerate() {
                    put(
                        &mut out.values,
                        &format!("model_vector_coefficient_{index}"),
                        x,
                    );
                }
                if (scale == 0 && rn != 0)
                    || (scale > 0 && rn > &scale * (Float::with_val(p, 1) >> (p / 2)))
                {
                    out.reason = Some(
                        "model vector residual unresolved; spectrum and counterfactual retained"
                            .into(),
                    );
                    out.outcome = "partial_unresolved".into();
                }
            } else {
                out.reason = Some("model vector lattice normalization unresolved".into());
                out.outcome = "partial_unresolved".into();
            }
        }
        Err(_) => {
            out.reason = Some(
                "model vector recovery unresolved; spectrum and counterfactual retained".into(),
            );
            out.outcome = "partial_unresolved".into();
        }
    }
    if let Some(err) = &f.tail_operator_error {
        put(
            &mut out.values,
            "declared_tail_form_error",
            &scalar(err, p)?,
        );
        put(
            &mut out.values,
            "conditional_energy_error_budget",
            &(scalar(err, p)? * norm2(&inv, p) * 2u32),
        );
    }
    for r in 0..n {
        let mut rr = row(r + 1, "generalized_model_eigenvalue");
        put(
            &mut rr.values,
            "energy",
            &(Float::with_val(p, &eigenvalues[r]) * 2u32),
        );
        for c in 0..n {
            put(
                &mut rr.values,
                &format!("zero_plus_tail_{c}"),
                &a[r * n + c],
            );
            put(&mut rr.values, &format!("tail_{c}"), &tail[r * n + c]);
            put(
                &mut rr.values,
                &format!("lattice_gram_{c}"),
                &gram[r * n + c],
            );
            put(
                &mut rr.values,
                &format!("whitened_operator_{c}"),
                &transformed[r * n + c],
            );
        }
        out.rows.push(rr);
    }
    out.convention="fixed supplied basis; (finite_zero+tail)*v=(E/2)*lattice_gram*v; no retained energy or band used to solve; conditional tail approximation scope remains external; no RH inference".into();
    Ok(out)
}
