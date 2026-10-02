//! Point diagnostics for complete retained-run research capture.
//! These producers do not change primary solves or establish infinite-limit claims.
use super::extended_research::{
    missing, put, report, row, unresolved, ExtendedAnalysis, ExtensionOptions,
    ExternalResearchInputs,
};
use super::retained_evidence::{dot, norm2, scalar, RetainedMatrix, RetainedRoots};
use super::state_geometry::RetainedState;
use anyhow::{bail, Result};
use rayon::prelude::*;
use rug::Float;
use serde::{Deserialize, Serialize};
use xc_cache::ContentDigest;

#[path = "convergence_capture/cluster_operator_math.rs"]
mod cluster_operator_math;
#[path = "convergence_capture/complex_math.rs"]
mod complex_math;
#[path = "convergence_capture/observation_math.rs"]
mod observation_math;
mod tail_error_math;
#[path = "convergence_capture/transfer_math.rs"]
mod transfer_math;
#[path = "convergence_capture/transport_math.rs"]
mod transport_math;
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
    /// Assumed spectral-norm bound on the additive tail form error in this basis.
    /// Its validity is external; the derived energy budget excludes numerical solve error.
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
        super::retained_evidence::precision(p)?;
        if dim == 0 || dim > 16385 || dim.is_multiple_of(2) {
            bail!("invalid retained source dimension");
        }
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
            for value in c
                .coefficients
                .iter()
                .chain(&c.matrix)
                .chain(std::iter::once(&c.eigenvalue))
            {
                scalar(value, c.precision_bits)?;
            }
            if scalar(&c.lambda_squared, c.precision_bits)? <= 1 {
                bail!("invalid comparison cutoff");
            }
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
fn vals(v: &[String], source_precision: u32, p: u32) -> Result<Vec<Float>> {
    v.iter()
        .map(|s| Ok(Float::with_val(p, scalar(s, source_precision)?)))
        .collect()
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
        "complex_transform" => complex_math::analyze(s, roots, o),
        "finite_section_transfer" => transfer_math::analyze(s, m, o, i),
        "observable_budget" => observation_math::analyze(s, roots, o, i),
        "operator_cluster" => cluster_operator_math::analyze(s, m, o, i),
        "tail_operator" => super::atom_research::tail_report(s, o, i),
        "root_transport" => transport_math::analyze(s, roots, o, i, directional),
        _ => bail!("unknown convergence diagnostic"),
    }
}
fn model_matvec(a: &[Float], v: &[Float], n: usize, p: u32) -> Result<Vec<Float>> {
    if n == 0 || n.checked_mul(n) != Some(a.len()) || v.len() != n {
        bail!("tail model matrix-vector shape mismatch");
    }
    a.par_chunks(n).map(|row| dot(row, v, p)).collect()
}
fn model_subtract_products(value: &Float, a: &[Float], b: &[Float], p: u32) -> Result<Float> {
    let mut x = Vec::with_capacity(a.len() + 1);
    x.push(value.clone());
    x.extend(a.iter().map(|v| -v.clone()));
    let mut y = Vec::with_capacity(b.len() + 1);
    y.push(Float::with_val(p, 1));
    y.extend_from_slice(b);
    dot(&x, &y, p)
}
pub(crate) fn tail_operator(
    s: &RetainedState,
    o: &ExtensionOptions,
    i: Option<&RunOnceInputs>,
    source_precision: u32,
) -> Result<ExtendedAnalysis> {
    super::retained_evidence::precision(source_precision)?;
    if source_precision > o.working_precision_bits {
        bail!("tail source precision exceeds working precision");
    }
    let store = super::capture_runtime::Checkpoints::new(&(
        "finite-tail-model-original-matrix-dense-source-recovery-v8",
        xc_numerics::eigen::STABLE_HOUSEHOLDER_SEMANTICS,
        xc_numerics::eigen::TRIDIAG_QR_SEMANTICS,
        xc_numerics::eigen::DENSE_EIGENVECTOR_SEMANTICS,
        source_precision,
        &s.manifest.content_digest,
        o,
        i,
    ))?;
    if let Some(result) = store.load::<ExtendedAnalysis>("model-solve")? {
        return Ok(result);
    }
    let result = tail_operator_uncached(s, o, i, source_precision)?;
    if result.outcome == "point_measurement" {
        if let Err(e) = store.save("model-solve", &result) {
            xc_core::progress_message!("tail model checkpoint unavailable: {e}");
        }
    }
    Ok(result)
}
fn tail_operator_uncached(
    s: &RetainedState,
    o: &ExtensionOptions,
    i: Option<&RunOnceInputs>,
    source_precision: u32,
) -> Result<ExtendedAnalysis> {
    let mut out = report("tail_operator", s, o);
    let Some(f) = i.and_then(|i| i.tail_form.as_ref()) else {
        return Ok(missing(
            out,
            "fixed-basis zero/lattice/tail forms required; model and omitted-tail formula are not inferred from a state",
        ));
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
    let finite = vals(&f.finite_zero_form, source_precision, p)?;
    let a = finite.clone();
    let tail = vals(&f.tail_correction, source_precision, p)?;
    let gram = vals(&f.lattice_gram, source_precision, p)?;
    let a = a
        .into_iter()
        .zip(&tail)
        .map(|(a, b)| a + b)
        .collect::<Vec<_>>();
    let mut pivots = Vec::with_capacity(n);
    let mut chol = vec![Float::with_val(p, 0); n * n];
    for r in 0..n {
        for c in 0..=r {
            let v = model_subtract_products(
                &gram[r * n + c],
                &chol[r * n..r * n + c],
                &chol[c * n..c * n + c],
                p,
            )?;
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
                chol[r * n + c] =
                    super::retained_evidence::point::quotient(&v, &chol[c * n + c], p)?;
            }
        }
    }
    let mut inv = vec![Float::with_val(p, 0); n * n];
    for c in 0..n {
        for r in c..n {
            let v = model_subtract_products(
                &Float::with_val(p, u32::from(r == c)),
                &chol[r * n + c..r * n + r],
                &(c..r).map(|k| inv[k * n + c].clone()).collect::<Vec<_>>(),
                p,
            )?;
            inv[r * n + c] = super::retained_evidence::point::quotient(&v, &chol[r * n + r], p)?;
        }
    }
    let whiten = |matrix: &[Float]| -> Result<Vec<Float>> {
        let left = (0..n)
            .into_par_iter()
            .map(|r| {
                (0..n)
                    .map(|c| {
                        dot(
                            &inv[r * n..(r + 1) * n],
                            &(0..n)
                                .map(|k| matrix[k * n + c].clone())
                                .collect::<Vec<_>>(),
                            p,
                        )
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let mut result = vec![Float::with_val(p, 0); n * n];
        for r in 0..n {
            for c in 0..=r {
                let value = dot(&left[r * n..(r + 1) * n], &inv[c * n..(c + 1) * n], p)?;
                result[r * n + c] = value.clone();
                result[c * n + r] = value;
            }
        }
        Ok(result)
    };
    let transformed = whiten(&a)?;
    let (diag, off, _) = xc_numerics::eigen::householder_tridiag_hp_stable(&transformed, n, p)?;
    let mut eigenvalues = xc_numerics::eigen::tridiag_eigenvalues_hp(&diag, &off, p)?;
    eigenvalues.sort_by(Float::total_cmp);
    let energy = Float::with_val(p, &eigenvalues[0]) * 2u32;
    put(&mut out.values, "model_energy", &energy);
    let retained = Float::with_val(p, scalar(&s.eigenvalue, s.precision)?);
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
        &norm2(&inv, p)?,
    );
    match xc_numerics::eigen::dense_symmetric_eigenvalues_hp_stable(&whiten(&finite)?, n, p) {
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
    // Recover from the original transformed source. A rounded Householder
    // similarity can lose tiny polynomial coefficients even when its vector
    // satisfies an absolute residual bound for the tridiagonal surrogate.
    let vector = xc_numerics::eigen::dense_symmetric_eigenvector_for_value_hp(
        &transformed,
        n,
        &eigenvalues[0],
        p,
        200,
    );
    match vector {
        Ok(y) => {
            let mut coeff = vec![Float::with_val(p, 0); n];
            for c in 0..n {
                coeff[c] = dot(
                    &(0..n).map(|r| inv[r * n + c].clone()).collect::<Vec<_>>(),
                    &y,
                    p,
                )?;
            }
            let gv = model_matvec(&gram, &coeff, n, p)?;
            let norm = dot(&coeff, &gv, p)?;
            if norm > 0 {
                for x in &mut coeff {
                    *x /= norm.clone().sqrt();
                }
                let av = model_matvec(&a, &coeff, n, p)?;
                let gv = model_matvec(&gram, &coeff, n, p)?;
                let residual = av
                    .iter()
                    .zip(&gv)
                    .map(|(a, g)| {
                        model_subtract_products(
                            a,
                            std::slice::from_ref(&eigenvalues[0]),
                            std::slice::from_ref(g),
                            p,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                let l2 = |v: &[Float]| v.iter().fold(Float::with_val(p, 0), |n, x| n.hypot(x));
                let rn = l2(&residual);
                let scale = l2(&av) + eigenvalues[0].clone().abs() * l2(&gv);
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
                    &dot(&coeff, &gv, p)?,
                );
                put(
                    &mut out.values,
                    "model_vector_zero_energy",
                    &(dot(&coeff, &model_matvec(&finite, &coeff, n, p)?, p)? * 2u32),
                );
                put(
                    &mut out.values,
                    "model_vector_tail_energy",
                    &(dot(&coeff, &model_matvec(&tail, &coeff, n, p)?, p)? * 2u32),
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
        let error = super::retained_evidence::finite_math::decimal_upper(err, p)?;
        out.values.insert(
            "declared_tail_form_error".into(),
            error.to_string_radix_round(10, None, rug::float::Round::Up),
        );
        match tail_error_math::bound(
            &gram,
            &error,
            n,
            p,
            o.maximum_working_bytes.unwrap_or(8 << 30),
        ) {
            Ok(bound) => {
                out.values.insert(
                    "conditional_energy_error_budget".into(),
                    bound.to_string_radix_round(10, None, rug::float::Round::Up),
                );
            }
            Err(error) => {
                out.outcome = "partial_unresolved".into();
                out.reason = Some(format!(
                    "conditional tail error unavailable: {error}; finite model retained"
                ));
            }
        }
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
    out.convention="declared source points; exact dyadic recipe forms rounded once; exact stored dot-product stages, point Cholesky/eigensolve; fixed supplied basis; (finite_zero+tail)*v=(E/2)*lattice_gram*v; no retained energy or band used to solve; conditional energy error bounds use exact stored Gram inverse trace and upward rounding, assume the declared tail spectral-norm error, and exclude numerical solve and source error; no RH inference".into();
    Ok(out)
}

#[cfg(test)]
mod exhaustive_resumed_validation_contract {
    use super::*;
    #[test]
    fn exhaustive_resumed_completion_checks_public_context() {
        for (dim, p) in [
            (0, 128),
            (2, 128),
            (16387, 128),
            (usize::MAX, 128),
            (1, 0),
            (1, 63),
            (1, 1_000_001),
        ] {
            assert!(
                super::super::research_completion::CompletionInputs::default()
                    .validate(dim, p)
                    .is_err(),
                "invalid completion context dim={dim}, p={p}"
            );
        }
    }
    #[test]
    fn exhaustive_resumed_run_once_checks_public_context() {
        let mut accepted = Vec::new();
        for (dim, p) in [
            (0, 128),
            (2, 128),
            (16387, 128),
            (usize::MAX, 128),
            (1, 0),
            (1, 63),
            (1, 1_000_001),
        ] {
            let result = std::panic::catch_unwind(|| RunOnceInputs::default().validate(dim, p));
            if !matches!(result, Ok(Err(_))) {
                accepted.push((dim, p));
            }
        }
        assert!(
            accepted.is_empty(),
            "invalid source contexts accepted: {accepted:?}"
        );
        assert!(RunOnceInputs::default().validate(1, 64).is_ok());
        assert!(RunOnceInputs::default().validate(16385, 1_000_000).is_ok());
    }
}
#[cfg(test)]
mod exhaustive_model_action {
    use super::*;
    #[test]
    fn exhaustive_model_action_preserves_cancelled_unit() {
        let p = 128;
        let x = Float::with_val(p, 1) << 400u32;
        let a = [
            x.clone(),
            Float::with_val(p, 1),
            -x,
            Float::with_val(p, 0),
            Float::with_val(p, 0),
            Float::with_val(p, 0),
            Float::with_val(p, 0),
            Float::with_val(p, 0),
            Float::with_val(p, 0),
        ];
        let got = model_matvec(&a, &vec![Float::with_val(p, 1); 3], 3, p).unwrap();
        assert_eq!(got[0], 1);
    }
}
