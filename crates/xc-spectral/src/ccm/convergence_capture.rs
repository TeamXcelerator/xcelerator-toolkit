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
/// Source-bound acquisition of the bounded diagnostic interfaces. Numerical
/// premises supplied by callers retain their own scope and are never promoted
/// to a statement about an unrepresented operator or function.
#[doc(hidden)]
pub mod finite_capture {
    use super::super::{
        extended_research::ExternalResearchInputs, retained_evidence::*,
        state_geometry::RetainedState,
    };
    use anyhow::{bail, Result};
    use rug::{Float, Rational};
    use serde::{Deserialize, Serialize};
    use serde_json::{json, Value};
    use std::collections::BTreeMap;
    use xc_cache::{
        ArtifactCacheContext, ArtifactExecutionCacheResult, ArtifactManifest, ContentDigest,
    };
    use xc_core::CancellationToken;
    use xc_solver::{
        convergence as cv,
        trial_energy::{self as te, streaming as stream},
        weighted_l1 as l1,
    };
    pub const KIND: &str = "ccm_finite_diagnostic_analysis";

    // Stream fingerprints computed in this process. A fingerprint reads only
    // the form metadata, the streaming options and the reader's exact entries,
    // so it is keyed by a digest of exactly those; failures are never kept.
    static FINGERPRINTS: std::sync::Mutex<Vec<(ContentDigest, stream::StreamRecord)>> =
        std::sync::Mutex::new(Vec::new());

    /// `compute()`, or the record of an identical earlier fingerprint.
    /// `content` must determine every entry the reader yields.
    fn fingerprint_once(
        metadata: &stream::FormMetadata,
        opts: &stream::Options,
        content: &ContentDigest,
        compute: impl FnOnce() -> Result<stream::StreamRecord>,
    ) -> Result<stream::StreamRecord> {
        let key = ContentDigest::sha256(&serde_json::to_vec(&(
            "stream-fingerprint-v1",
            metadata,
            opts,
            content,
        ))?);
        let retained = FINGERPRINTS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .find(|(retained, _)| retained == &key)
            .map(|(_, record)| record.clone());
        if let Some(record) = retained {
            return Ok(record);
        }
        let record = compute()?;
        let mut retained = FINGERPRINTS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if retained.len() >= 16 {
            retained.remove(0);
        }
        retained.push((key, record.clone()));
        Ok(record)
    }

    /// Exact digest of the stored entries `MatrixReader` reads: precision,
    /// sign and exact value of every scalar, in row-major order.
    pub(crate) fn matrix_content(m: &RetainedMatrix<'_>, complex: bool) -> ContentDigest {
        use sha2::{Digest, Sha256};
        let mut hash = Sha256::new();
        hash.update(b"retained-matrix-reader-entries-v1\0");
        hash.update((m.modes as u64).to_le_bytes());
        hash.update([u8::from(complex)]);
        hash.update((m.entries.len() as u64).to_le_bytes());
        for x in m.entries.iter() {
            hash.update(x.prec().to_le_bytes());
            hash.update([u8::from(x.is_sign_negative())]);
            match x.to_integer_exp() {
                Some((significand, exponent)) => {
                    let digits = significand.to_digits::<u8>(rug::integer::Order::Lsf);
                    hash.update([1]);
                    hash.update(exponent.to_le_bytes());
                    hash.update((digits.len() as u64).to_le_bytes());
                    hash.update(&digits);
                }
                None => hash.update([if x.is_nan() { 2 } else { 3 }]),
            }
        }
        ContentDigest(format!("{:x}", hash.finalize()))
    }
    /// A computation stopped by a resource limit: reported as blocked and never
    /// retained, so a later run with more resources recomputes it.
    #[derive(Debug)]
    struct ResourceLimited(String);
    impl std::fmt::Display for ResourceLimited {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(&self.0)
        }
    }
    impl std::error::Error for ResourceLimited {}
    pub const SEMANTICS: &str = "ccm-finite-diagnostic-capture-v6";
    const PROFILE_RECIPE: &str = "paired_center_and_bounded_functional_complex_profiles_v3";
    const BASIS: &str = "centered_full_V_fourier";
    const NORMALIZATION: &str =
        "raw_coefficients; quadratic_form_sym_Tau; metric_identity; coefficient_l2";

    fn prolate_convention() -> Value {
        json!({"convention":"D=-log(deficit)","logarithm":"natural"})
    }

    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct Vector {
        pub label: String,
        pub coefficients: Vec<te::ExactBounds>,
    }
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct TrialSeries {
        pub basis_id: String,
        /// Domain, projection, stage construction and approximation errors.
        /// The adapter does not infer these from a vector's name.
        pub scope: String,
        pub baseline: Vector,
        pub corrections: Vec<Vector>,
        #[serde(default)]
        pub provenance: BTreeMap<String, String>,
    }
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct ComplexVector {
        pub label: String,
        pub real: Vec<te::ExactBounds>,
        pub imaginary: Vec<te::ExactBounds>,
    }
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct ComplexTrialSeries {
        pub basis_id: String,
        pub matrix_digest: ContentDigest,
        pub scope: String,
        pub baseline: ComplexVector,
        pub corrections: Vec<ComplexVector>,
        #[serde(default)]
        pub functional: Option<BoundedFunctional>,
        #[serde(default)]
        pub provenance: BTreeMap<String, String>,
    }
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
    pub enum FunctionalDenominator {
        ReferenceSquaredNorm,
        Declared {
            value: te::ExactBounds,
            provenance: String,
            scope: String,
        },
    }
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct BoundedFunctional {
        pub id: String,
        /// Finite coefficient identity metric on the declared Fourier window.
        pub reference: ComplexVector,
        pub denominator: FunctionalDenominator,
    }
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct ProfileRequest {
        pub basis_id: String,
        pub lambda_squared: String,
        pub scope: String,
        pub definition_digest: ContentDigest,
        pub profiles: Vec<ComplexVector>,
        pub functional: BoundedFunctional,
        pub intervals_per_half: usize,
        pub precision_bits: u32,
    }
    #[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct Inputs {
        pub scope: String,
        #[serde(default)]
        pub l1: Option<l1::WeightedL1Problem>,
        #[serde(default)]
        pub l1_options: l1::WeightedL1Options,
        #[serde(default)]
        pub trials: Option<TrialSeries>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub complex_trials: Option<ComplexTrialSeries>,
        /// Independent references supplied by the provider. No eigenstate is
        /// invented or needed to evaluate this target-only request.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub target_profiles: Option<ProfileRequest>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub projection_energy: Option<energy_extensions::ProjectionRequest>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub component_energy: Option<energy_extensions::ComponentRequest>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub continuous_energy: Option<energy_extensions::ContinuousRequest>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub refinement_cohort: Option<RefinementCohort>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub energy_distance: Option<EnergyDistancePremises>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        pub matched_spectral_triples: Vec<[RefinementObservation; 3]>,
        #[serde(default)]
        pub streaming_options: stream::Options,
        #[serde(default)]
        pub convergence: BTreeMap<String, cv::Problem>,
        #[serde(default)]
        pub convergence_options: cv::Options,
        /// A diagnostic can be explicitly inapplicable, with a retained reason.
        /// Such declarations cannot override a supplied calculation request.
        #[serde(default)]
        pub not_applicable: BTreeMap<String, String>,
        #[serde(default)]
        pub prolate_references: Vec<ProlateReference>,
    }
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct ProlateReference {
        pub full_index: u32,
        pub source_id: String,
        /// Natural log of the numerical singular-value deficit 1-|chi_n|.
        /// Concentration deficits must be converted by the supplying pipeline.
        pub log_singular_value_deficit: String,
        pub precision_bits: u32,
        pub quadrature_order: usize,
        pub scope: String,
        /// An explicitly paired observation, never an inferred ordinal join.
        pub weil: Option<WeilReference>,
    }
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct WeilReference {
        pub source_id: String,
        pub sector: String,
        pub zero_based_index: usize,
        pub sector_dimension: usize,
        pub log_absolute_eigenvalue: String,
    }
    fn problem_id(p: &cv::Problem) -> &'static str {
        match p {
            cv::Problem::Root(_) => "finite_root_budget",
            cv::Problem::Directional(_) => "directional_error_bound",
            cv::Problem::TailBlock(_) => "finite_tail_bound",
            cv::Problem::Budget(_) => "dimension_precision_budget",
            cv::Problem::Normalization(_) => "normalization_error_bound",
            cv::Problem::Profile(_) => "continuous_l1_bound",
            cv::Problem::Cluster(_) => "spectral_cluster_bound",
        }
    }
    impl Inputs {
        pub fn validate(&self) -> Result<()> {
            if self.scope.trim().is_empty()
                || self.scope.len() > 16384
                || serde_json::to_vec(self)?.len() as u64
                    > crate::ccm::capture_runtime::RESEARCH_INPUT_MAXIMUM_BYTES
            {
                bail!("finite diagnostic input scope or byte limit");
            }
            for (id, p) in &self.convergence {
                if id != problem_id(p) {
                    bail!("finite diagnostic problem/identifier mismatch");
                }
            }
            if self.prolate_references.len() > 64 {
                bail!("indexed reference row limit");
            }
            if self.matched_spectral_triples.len() > 64 {
                bail!("matched spectral triple row limit");
            }
            for r in &self.prolate_references {
                if r.full_index > 64
                    || !r.full_index.is_multiple_of(2)
                    || r.source_id.is_empty()
                    || r.scope.is_empty()
                    || !(64..=65536).contains(&r.precision_bits)
                    || r.quadrature_order == 0
                {
                    bail!("invalid indexed reference identity, precision or quadrature");
                }
                rational(&r.log_singular_value_deficit)?;
                if let Some(w) = &r.weil {
                    if w.source_id.is_empty()
                        || w.sector != "even"
                        || !r.full_index.is_multiple_of(4)
                        || w.sector_dimension == 0
                        || w.sector_dimension > 16385
                        || w.zero_based_index >= w.sector_dimension
                    {
                        bail!("invalid explicit Weil reference join");
                    }
                    rational(&w.log_absolute_eigenvalue)?;
                }
            }
            for (id, reason) in &self.not_applicable {
                if !super::super::capture::FINITE_DIAGNOSTICS.contains(&id.as_str())
                    || reason.trim().is_empty()
                    || reason.len() > 16384
                    || self.convergence.contains_key(id)
                    || (id == "constrained_l1_fit" && self.l1.is_some())
                    || (id == "continuous_l1_bound" && self.target_profiles.is_some())
                    || (id == "dimension_precision_budget" && self.refinement_cohort.is_some())
                    || (id == "spectral_cluster_bound" && !self.matched_spectral_triples.is_empty())
                    || (matches!(id.as_str(), "trial_vector_energy" | "trial_vector_parity")
                        && self.energy_distance.is_some())
                    || (matches!(id.as_str(), "trial_vector_energy" | "trial_vector_parity")
                        && self.trials.is_some())
                {
                    bail!("invalid or conflicting diagnostic applicability");
                }
            }
            if let Some(t) = &self.trials {
                if t.basis_id != BASIS
                    || t.scope.trim().is_empty()
                    || t.scope.len() > 16384
                    || t.corrections.len() > 15
                    || t.provenance.len() > 64
                {
                    bail!("trial series basis, scope or part limit");
                }
            }
            if let Some(t) = &self.complex_trials {
                if self.trials.is_some()
                    || t.basis_id != BASIS
                    || !t.matrix_digest.validate()
                    || t.scope.trim().is_empty()
                    || t.scope.len() > 16384
                    || t.corrections.len() > 15
                    || t.provenance.len() > 64
                    || self.not_applicable.contains_key("trial_vector_energy")
                    || self.not_applicable.contains_key("trial_vector_parity")
                {
                    bail!("complex trial series identity, applicability or part limit");
                }
            }
            if self.refinement_cohort.is_some()
                && self.convergence.contains_key("dimension_precision_budget")
            {
                bail!("choose either a supplied analytic budget or a refinement cohort for this diagnostic");
            }
            for (present, group) in [
                (self.projection_energy.is_some(), "finite_tail_bound"),
                (self.component_energy.is_some(), "trial_vector_energy"),
                (self.continuous_energy.is_some(), "trial_vector_energy"),
            ] {
                if present && self.not_applicable.contains_key(group) {
                    bail!("conflicting supplemental energy applicability");
                }
            }
            xc_core::validate_secret_free(self, "finite diagnostic inputs")?;
            Ok(())
        }
    }

    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct Row {
        pub label: String,
        pub outcome: String,
        pub result: Value,
    }
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct Analysis {
        pub diagnostic: String,
        pub outcome: String,
        pub reason: Option<String>,
        pub scope: String,
        pub source_precision_bits: u32,
        pub rows: Vec<Row>,
        pub result: Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub input_preparation_error: Option<String>,
    }
    fn empty(id: &str, s: &RetainedState, status: &str, reason: &str) -> Analysis {
        Analysis { diagnostic: id.into(), outcome: status.into(), reason: Some(reason.into()),
            scope: "finite stored inputs only; caller analytic premises remain conditional; no continuum or global ordinal inference".into(),
            source_precision_bits: s.precision, rows: vec![], result: Value::Null, input_preparation_error: None }
    }
    fn computed(id: &str, s: &RetainedState, result: Value, outcome: &str) -> Analysis {
        let mut a = empty(id, s, "computed", "");
        a.reason = None;
        a.rows.push(Row {
            label: id.into(),
            outcome: outcome.into(),
            result: Value::Null,
        });
        a.result = result;
        a
    }
    fn float_bits(x: &Float) -> Result<(u64, u64)> {
        if !x.is_finite() {
            bail!("nonfinite retained scalar");
        }
        if x.is_zero() {
            return Ok((1, 0));
        }
        let e = i64::from(x.get_exp().unwrap());
        let numerator = e.max(i64::from(x.prec())).max(1) as u64;
        let denominator = (i64::from(x.prec()) - e).max(0) as u64;
        if numerator > 65536 || denominator > 65536 {
            bail!("retained scalar exceeds exact rational bit limit before conversion");
        }
        Ok((numerator, denominator))
    }
    #[test]
    fn finite_scalar_admission_checks_exponent_before_allocation() {
        for exponent in [1000000, -1000000] {
            let x = Float::with_val(128, 1) << exponent;
            assert!(float_bits(&x).is_err());
            assert!(exact(&x).is_err());
        }
        assert_eq!(exact(&Float::with_val(128, 0)).unwrap(), 0);
        assert_eq!(exact(&Float::with_val(128, 1)).unwrap(), 1);
    }
    fn exact(x: &Float) -> Result<Rational> {
        float_bits(x)?;
        x.to_rational()
            .ok_or_else(|| anyhow::anyhow!("nonfinite retained scalar"))
    }
    fn point(x: Rational) -> te::ExactBounds {
        te::ExactBounds::point(x.to_string())
    }
    fn rational(text: &str) -> Result<Rational> {
        use rug::ops::Pow;
        if text.is_empty() || text.len() > 131072 || text.bytes().any(|b| b.is_ascii_whitespace()) {
            bail!("exact scalar size or syntax");
        }
        let x = if text.contains('/') {
            Rational::from_str_radix(text, 10)?
        } else {
            let d = xc_core::DecimalLiteral::new(text)?.canonical()?;
            let (m, e) = d.as_str().split_once('e').unwrap_or((d.as_str(), "0"));
            let e: i32 = e.parse()?;
            if e.unsigned_abs() > 16384 {
                bail!("decimal exponent limit");
            }
            let scale = rug::Integer::from(10).pow(e.unsigned_abs());
            let m = rug::Integer::from_str_radix(m, 10)?;
            if e >= 0 {
                Rational::from(m * scale)
            } else {
                Rational::from((m, scale))
            }
        };
        if x.numer().significant_bits() > 65536 || x.denom().significant_bits() > 65536 {
            bail!("exact scalar bit limit");
        }
        Ok(x)
    }

    type I = xc_numerics::interval::RationalInterval;
    type C = (I, I);
    fn read_bound(x: &te::ExactBounds) -> Result<I> {
        Ok(I::new(rational(&x.lower)?, rational(&x.upper)?)?)
    }
    fn pack_bound(x: &I) -> te::ExactBounds {
        te::ExactBounds {
            lower: x.lower().to_string(),
            upper: x.upper().to_string(),
        }
    }
    fn checked_interval(x: I) -> Result<I> {
        for v in [x.lower(), x.upper()] {
            if v.numer().significant_bits() > 65536 || v.denom().significant_bits() > 65536 {
                bail!("finite profile rational work budget exceeded");
            }
        }
        Ok(x)
    }
    fn complex_coefficients(v: &ComplexVector, n: usize) -> Result<Vec<C>> {
        if v.label.trim().is_empty()
            || v.label.len() > 256
            || n == 0
            || n > 16385
            || v.real.len() != n
            || v.imaginary.len() != n
        {
            bail!("complex vector identity or dimension");
        }
        v.real
            .iter()
            .zip(&v.imaginary)
            .map(|(r, i)| Ok((read_bound(r)?, read_bound(i)?)))
            .collect()
    }
    fn dot(a: &[C], b: &[C]) -> Result<C> {
        if a.len() != b.len() {
            bail!("functional vector dimension mismatch");
        }
        let mut r = I::point(Rational::from(0));
        let mut i = r.clone();
        for ((ar, ai), (br, bi)) in a.iter().zip(b) {
            r = checked_interval(r.add(&ar.mul(br)).add(&ai.mul(bi)))?;
            i = checked_interval(i.add(&ar.mul(bi)).sub(&ai.mul(br)))?;
        }
        Ok((r, i))
    }
    fn squared_norm(a: &[C]) -> Result<I> {
        a.iter().try_fold(I::point(Rational::from(0)), |s, (r, i)| {
            checked_interval(s.add(&r.square()).add(&i.square()))
        })
    }
    fn abs_squared(z: &C) -> I {
        z.0.square().add(&z.1.square())
    }
    fn profile_interval(x: I, precision: u32) -> Result<I> {
        use rug::float::Round;
        // Exact complex division can introduce different large odd denominators
        // at each interval endpoint. Subsequent norm sums multiply those factors.
        // Dyadic outward enclosures keep that representation growth bounded while
        // including both the original uncertainty and the rounding error.
        let lower = Float::with_val_round(precision, x.lower(), Round::Down).0;
        let upper = Float::with_val_round(precision, x.upper(), Round::Up).0;
        checked_interval(I::new(exact(&lower)?, exact(&upper)?)?)
    }
    fn complex_divide(z: &C, a: &C, precision: u32) -> Result<C> {
        let d = abs_squared(a);
        Ok((
            profile_interval(z.0.mul(&a.0).add(&z.1.mul(&a.1)).div(&d)?, precision)?,
            profile_interval(z.1.mul(&a.0).sub(&z.0.mul(&a.1)).div(&d)?, precision)?,
        ))
    }
    fn pack_complex(z: &C) -> Value {
        json!({"real":pack_bound(&z.0),"imaginary":pack_bound(&z.1)})
    }
    fn functional_denominator(f: &BoundedFunctional, reference: &[C]) -> Result<I> {
        if f.id.trim().is_empty() || f.id.len() > 1024 {
            bail!("functional identity missing");
        }
        match &f.denominator {
            FunctionalDenominator::ReferenceSquaredNorm => squared_norm(reference),
            FunctionalDenominator::Declared {
                value,
                provenance,
                scope,
            } => {
                if provenance.trim().is_empty()
                    || scope.trim().is_empty()
                    || provenance.len() > 16384
                    || scope.len() > 16384
                {
                    bail!("declared functional denominator requires scope and provenance");
                }
                let value = read_bound(value)?;
                if value.lower() < &0 {
                    bail!("functional norm denominator must be nonnegative");
                }
                Ok(value)
            }
        }
    }
    fn functional_value(reference: &[C], denominator: &I, v: &[C]) -> Result<C> {
        let z = dot(reference, v)?;
        Ok((
            checked_interval(z.0.div(denominator)?)?,
            checked_interval(z.1.div(denominator)?)?,
        ))
    }
    fn sqrt_bound(x: &I, p: u32) -> Result<I> {
        use rug::float::Round;
        let mut lo = Float::with_val_round(p, x.lower(), Round::Down).0;
        let mut hi = Float::with_val_round(p, x.upper(), Round::Up).0;
        lo.sqrt_round(Round::Down);
        hi.sqrt_round(Round::Up);
        Ok(I::new(exact(&lo)?, exact(&hi)?)?)
    }
    fn functional_stage_energy(
        f: &BoundedFunctional,
        series: &TrialSeries,
        n: usize,
        report: &stream::Report,
    ) -> Result<Value> {
        let reference = complex_coefficients(&f.reference, n)?;
        let denominator = functional_denominator(f, &reference)?;
        if !denominator.is_strictly_positive() {
            return Ok(
                json!({"status":"unresolved","reason":"functional denominator includes zero","functional":f}),
            );
        }
        let mut sum = vec![(I::point(Rational::from(0)), I::point(Rational::from(0))); n];
        let mut rows = vec![];
        for (part, stage) in std::iter::once(&series.baseline)
            .chain(&series.corrections)
            .zip(&report.stages)
        {
            for (j, z) in sum.iter_mut().enumerate() {
                z.0 = checked_interval(z.0.add(&read_bound(&part.coefficients[j])?))?;
                if part.coefficients.len() == 2 * n {
                    z.1 = checked_interval(z.1.add(&read_bound(&part.coefficients[n + j])?))?;
                }
            }
            let value = functional_value(&reference, &denominator, &sum)?;
            let d = abs_squared(&value);
            rows.push(if d.is_strictly_positive() {
                json!({"corrections_applied":stage.corrections_applied,"status":"finite_enclosure",
                    "functional_value":pack_complex(&value),"functional_magnitude_floor":sqrt_bound(&d,256)?.lower().to_string(),
                    "energy_over_functional_squared":pack_bound(&read_bound(&stage.measurement.energy)?.div(&d)?),
                    "squared_norm_over_functional_squared":pack_bound(&read_bound(&stage.measurement.squared_norm)?.div(&d)?)})
            } else {
                json!({"corrections_applied":stage.corrections_applied,"status":"unresolved","reason":"functional value includes zero","functional_value":pack_complex(&value)})
            });
        }
        Ok(
            json!({"functional":f,"rows":rows,"normalization":"complex division by P(v); energy Q(v)/|P(v)|^2; distinct from Rayleigh quotient",
            "scope":"finite coefficient identity metric; declared external denominator remains a supplied premise"}),
        )
    }

    /// Independent finite Fourier profile analysis. Only the supplied numerical
    /// functions enter this calculation; no primary CCM solve is available here.
    #[doc(hidden)]
    pub fn analyze_profiles(request: &ProfileRequest) -> Result<Value> {
        analyze_profiles_inner(request, true)
    }
    fn analyze_profiles_inner(request: &ProfileRequest, include_samples: bool) -> Result<Value> {
        use rayon::prelude::*;
        use rug::float::Round;
        use xc_numerics::mpfr_interval::MpfrInterval as M;
        let n = request.profiles.first().map_or(0, |v| v.real.len());
        let p = request.precision_bits;
        let q = request.intervals_per_half;
        if request.basis_id != BASIS
            || n == 0
            || n.is_multiple_of(2)
            || n > 8193
            || !(64..=8192).contains(&p)
            || !(8..=8192).contains(&q)
            || !(2..=18).contains(&request.profiles.len())
            || request.scope.trim().is_empty()
            || request.scope.len() > 16384
            || !request.definition_digest.validate()
            || n as u128
                * q as u128
                * request.profiles.len() as u128
                * request.profiles.len().saturating_sub(1) as u128
                > 8_000_000
            || serde_json::to_vec(request)?.len() > 64 * 1024 * 1024
        {
            bail!("profile request identity or workspace limit");
        }
        xc_core::validate_secret_free(request, "profile request")?;
        xc_numerics::mpfr_interval::ensure_uniform_exponent_range()?;
        let cutoff = rational(&request.lambda_squared)?;
        if cutoff <= 1 {
            bail!("profile cutoff must exceed one");
        }
        let mut labels = std::collections::BTreeSet::new();
        let vectors = request
            .profiles
            .iter()
            .map(|v| {
                if !labels.insert(&v.label) {
                    bail!("duplicate profile label");
                }
                complex_coefficients(v, n)
            })
            .collect::<Result<Vec<_>>>()?;
        let reference = complex_coefficients(&request.functional.reference, n)?;
        let denominator = functional_denominator(&request.functional, &reference)?;
        let convert = |x: &I| -> Result<M> {
            Ok(M::new(
                Float::with_val_round(p, x.lower(), Round::Down).0,
                Float::with_val_round(p, x.upper(), Round::Up).0,
            )?)
        };
        let pack = |x: &M| -> Result<te::ExactBounds> {
            x.validate()?;
            Ok(te::ExactBounds {
                lower: exact(x.lower())?.to_string(),
                upper: exact(x.upper())?.to_string(),
            })
        };
        let center = |v: &[C]| -> Result<C> {
            let mut z = (I::point(Rational::from(0)), I::point(Rational::from(0)));
            for (j, (r, i)) in v.iter().enumerate() {
                let sign = I::point(Rational::from(if j.abs_diff(n / 2).is_multiple_of(2) {
                    1
                } else {
                    -1
                }));
                z.0 = checked_interval(z.0.add(&r.mul(&sign)))?;
                z.1 = checked_interval(z.1.add(&i.mul(&sign)))?;
            }
            Ok(z)
        };
        let log_c = M::from_rational(&cutoff, p).ln()?;
        let zero = || M::from_i64(0, p);
        let mut normalizers = vec![];
        let mut normalized = vec![];
        for (v, original) in vectors.iter().zip(&request.profiles) {
            let a = center(v)?;
            let b = if denominator.is_strictly_positive() {
                Some(functional_value(&reference, &denominator, v)?)
            } else {
                None
            };
            let ratio = b
                .as_ref()
                .filter(|_| abs_squared(&a).is_strictly_positive())
                .map(|b| complex_divide(b, &a, p))
                .transpose()?;
            normalizers.push(json!({"label":original.label,"center":pack_complex(&a),"functional":b.as_ref().map(pack_complex),
                "reciprocal_even_point_coefficients":v.iter().all(|z|z.0.is_point() && z.1.is_point()) && v.iter().eq(v.iter().rev()),
                "center_magnitude_floor":sqrt_bound(&abs_squared(&a),p)?.lower().to_string(),
                "functional_magnitude_floor":b.as_ref().map(|z|sqrt_bound(&abs_squared(z),p).map(|v|v.lower().to_string())).transpose()?,
                "functional_over_center":ratio.as_ref().map(pack_complex),"raw_coefficient_squared_norm":pack_bound(&squared_norm(v)?)}));
            normalized.push(
                [Some(a), b]
                    .iter()
                    .map(|value| -> Result<Option<Vec<C>>> {
                        value
                            .as_ref()
                            .filter(|a| abs_squared(a).is_strictly_positive())
                            .map(|a| {
                                v.iter()
                                    .map(|z| complex_divide(z, a, p))
                                    .collect::<Result<Vec<_>>>()
                            })
                            .transpose()
                    })
                    .collect::<Result<Vec<_>>>()?,
            );
        }
        // Evaluate every node once for each representation, retaining both
        // reciprocal halves. These are directed function values, not quadrature
        // error estimates. The interval-range integral below has separate scope.
        let evaluate = |v: &[C], t: &M| -> Result<(M, M)> {
            let angle = M::pi(p).mul(&t.add(&M::from_i64(1, p)));
            let mut re = zero();
            let mut im = zero();
            for (j, (r, i)) in v.iter().enumerate() {
                let phase = angle.mul(&M::from_i64(j as i64 - (n / 2) as i64, p));
                let (c, s) = (phase.cos(), phase.sin());
                let (r, i) = (convert(r)?, convert(i)?);
                re = re.add(&r.mul(&c)).sub(&i.mul(&s));
                im = im.add(&r.mul(&s)).add(&i.mul(&c));
            }
            re.validate()?;
            im.validate()?;
            Ok((re, im))
        };
        let weight = |t: &M| -> Result<M> {
            Ok(log_c
                .div(&M::from_i64(2, p))?
                .mul(&log_c.mul(t).div(&M::from_i64(4, p))?.exp()))
        };
        // Pairs are independent; they are evaluated concurrently and kept in
        // the serial (normalization, left, right) order, as is the first failure.
        let pair = |mode: usize, name: &str, left: usize, right: usize| -> Result<Value> {
            let (Some(a), Some(b)) = (&normalized[left][mode], &normalized[right][mode]) else {
                return Ok(
                    json!({"left":request.profiles[left].label,"right":request.profiles[right].label,"normalization":name,"status":"unresolved","reason":"normalizer includes zero"}),
                );
            };
            let delta = a
                .iter()
                .zip(b)
                .map(|((ar, ai), (br, bi))| {
                    Ok((checked_interval(ar.sub(br))?, checked_interval(ai.sub(bi))?))
                })
                .collect::<Result<Vec<_>>>()?;
            let coefficient_l2 = sqrt_bound(&squared_norm(&delta)?, p)?;
            let raw_delta = vectors[left]
                .iter()
                .zip(&vectors[right])
                .map(|((ar, ai), (br, bi))| {
                    Ok((checked_interval(ar.sub(br))?, checked_interval(ai.sub(bi))?))
                })
                .collect::<Result<Vec<_>>>()?;
            let raw_error = sqrt_bound(&squared_norm(&raw_delta)?, p)?;
            let reference_norm = sqrt_bound(&squared_norm(&vectors[right])?, p)?;
            let na = if mode == 0 {
                center(&vectors[left])?
            } else {
                functional_value(&reference, &denominator, &vectors[left])?
            };
            let nb = if mode == 0 {
                center(&vectors[right])?
            } else {
                functional_value(&reference, &denominator, &vectors[right])?
            };
            let af = sqrt_bound(&abs_squared(&na), p)?;
            let bf = sqrt_bound(&abs_squared(&nb), p)?;
            let diff = sqrt_bound(&abs_squared(&(na.0.sub(&nb.0), na.1.sub(&nb.1))), p)?;
            let propagated = if af.lower() > &0 && bf.lower() > &0 {
                Some(
                    I::point(raw_error.upper().clone())
                        .div(&I::point(af.lower().clone()))?
                        .add(
                            &I::point(reference_norm.upper().clone())
                                .mul(&I::point(diff.upper().clone()))
                                .div(&I::point(Rational::from(af.lower() * bf.lower())))?,
                        ),
                )
            } else {
                None
            };
            let real_normalization_witness =
                if na.1 == I::point(Rational::from(0)) && nb.1 == I::point(Rational::from(0)) {
                    let problem = cv::Problem::Normalization(cv::NormalizationProblem {
                        source_id: format!("{}:{}:{}", request.definition_digest.0, left, right),
                        raw_norm_error_upper: raw_error.upper().to_string(),
                        reference_norm_upper: reference_norm.upper().to_string(),
                        source_normalizer: pack_bound(&na.0),
                        reference_normalizer: pack_bound(&nb.0),
                        normalizer_difference_upper: diff.upper().to_string(),
                    });
                    let options = cv::Options {
                        working_precision_bits: p,
                        maximum_rational_bits: 65536,
                        ..Default::default()
                    };
                    Some(cv::analyze(&problem, &options, &CancellationToken::new())?)
                } else {
                    None
                };
            let mut halves = vec![];
            for half in [-1i64, 1].into_iter().filter(|_| include_samples) {
                let mut samples = vec![];
                let mut quadrature = zero();
                let mut integral = zero();
                // Nodes are independent; their terms are accumulated
                // below in node order, as in the serial loop.
                let nodes = (0..=q)
                    .into_par_iter()
                    .map(|j| -> Result<(M, Value, Option<M>)> {
                        let t = Rational::from((half * j as i64, q as i64));
                        let tm = M::from_rational(&t, p);
                        let (r, i) = evaluate(&delta, &tm)?;
                        let norm = r.square().add(&i.square()).sqrt()?;
                        let w = weight(&tm)?;
                        let quadrature_term = norm.mul(&w).div(&M::from_u64(
                            if j == 0 || j == q {
                                2 * q as u64
                            } else {
                                q as u64
                            },
                            p,
                        ))?;
                        let sample =
                            json!({"t":t.to_string(),"real":pack(&r)?,"imaginary":pack(&i)?});
                        let integral_term = if j < q {
                            let next = Rational::from((half * (j + 1) as i64, q as i64));
                            let cell = I::hull(t, next);
                            let tm = convert(&cell)?;
                            let (r, i) = evaluate(&delta, &tm)?;
                            let magnitude = r.square().add(&i.square()).sqrt()?;
                            Some(
                                magnitude
                                    .mul(&weight(&tm)?)
                                    .div(&M::from_u64(q as u64, p))?,
                            )
                        } else {
                            None
                        };
                        Ok((quadrature_term, sample, integral_term))
                    })
                    .collect::<Vec<_>>();
                for node in nodes {
                    let (quadrature_term, sample, integral_term) = node?;
                    quadrature = quadrature.add(&quadrature_term);
                    samples.push(sample);
                    if let Some(term) = integral_term {
                        integral = integral.add(&term);
                    }
                }
                halves.push(json!({"half":if half>0 {"u>=1"} else {"u<=1"},"signed_residual_samples":samples,
                        "trapezoid_value_enclosure":pack(&quadrature)?,"continuous_weighted_l1_enclosure":pack(&integral)?}));
            }
            Ok(
                json!({"left":request.profiles[left].label,"right":request.profiles[right].label,"normalization":name,"status":"finite_enclosure",
                    "coefficient_l2_difference":pack_bound(&coefficient_l2),"propagated_norm_error_upper":propagated.as_ref().map(|v|v.upper().to_string()),
                    "real_normalization_witness":real_normalization_witness,"halves":halves}),
            )
        };
        let count = vectors.len();
        let order = ["center", "bounded_functional"]
            .into_iter()
            .enumerate()
            .flat_map(|(mode, name)| {
                (0..count).flat_map(move |left| {
                    (left + 1..count).map(move |right| (mode, name, left, right))
                })
            })
            .collect::<Vec<_>>();
        let mut pairs = vec![];
        for value in order
            .into_par_iter()
            .map(|(mode, name, left, right)| pair(mode, name, left, right))
            .collect::<Vec<_>>()
        {
            pairs.push(value?);
        }
        let weighted_l1_dual = if denominator.is_strictly_positive() {
            let sup = reference.iter().try_fold(zero(), |sum, z| -> Result<M> {
                Ok(sum.add(&convert(&abs_squared(z))?.sqrt()?))
            })?;
            let base = sup.div(&log_c.mul(&convert(&denominator)?))?;
            Some(
                json!({"full_window_upper":pack(&base.mul(&log_c.div(&M::from_i64(4,p))?.exp()))?.upper,
                "positive_half_upper_on_reciprocal_even_arguments":pack(&base.mul(&M::from_i64(2,p)))?.upper,
                "scope":"P uses the normalized full-window log-measure inner product; dual constants refer to weighted L1 with u^-1/2 du; the positive-half constant requires a reciprocal-even argument, not merely a small odd component"}),
            )
        } else {
            None
        };
        let bound = if denominator.is_strictly_positive() {
            Some(pack_bound(
                &sqrt_bound(&squared_norm(&reference)?, p)?.div(&denominator)?,
            ))
        } else {
            None
        };
        Ok(
            json!({"semantics":"finite-paired-profile-normalization-v1","request_digest":ContentDigest::sha256(&serde_json::to_vec(request)?),
            "scope":"finite stored Fourier functions only; excludes projection, target approximation and exterior tails; interval range integration encloses the finite-function integral; trapezoid enclosure covers point arithmetic only",
            "coordinate":"t=2 log(u)/log(C); Fourier exp(i*pi*j*(t+1)); weighted L1 measure u^-1/2 du",
            "metric":"coefficient identity, equal to normalized full-window Fourier L2 metric",
            "ccm_orthonormal_coefficient_scale_squared":pack(&log_c)?,
            "orthonormal_ccm_basis_conversion":"these profiles use exp(i*pi*j*(t+1)); to represent the same functions in V_j=exp(i*pi*j*(t+1))/sqrt(log(C)), multiply coefficients by sqrt(log(C)); both the raw quadratic form and squared norm multiply by log(C), while Rayleigh quotients are unchanged",
            "functional_convention":"P(v)=<reference,v>/denominator; complex-linear in v; division by P is not a least-squares fit",
            "normalization_arithmetic":"complex coefficient quotients enclosed by outward-rounded dyadic endpoints at precision_bits before norm accumulation",
            "functional":request.functional,"denominator":pack_bound(&denominator),"functional_operator_norm_bound":bound,
            "functional_weighted_l1_dual_bounds":weighted_l1_dual,
            "normalizers":normalizers,"pairs":pairs,"precision_bits":p,"intervals_per_half":q,
            "samples_included":include_samples,
            "triangle_scope":"pairwise distances retained separately; triangle inequality is an upper bound, not an additive identity"}),
        )
    }
    #[doc(hidden)]
    pub fn capture_target_profiles(
        request: &ProfileRequest,
        cache: &ArtifactCacheContext<'_>,
    ) -> Result<ArtifactExecutionCacheResult<ResearchRecord<Value>>> {
        if cache.write_visibility == xc_cache::CacheVisibility::Public {
            bail!("target profile capture is private only");
        }
        // Full request is retained so replay needs neither an eigenstate nor an
        // executable target provider. The usual private evidence family applies.
        managed(
            KIND,
            json!({"semantics":SEMANTICS,"profile_recipe":PROFILE_RECIPE,"diagnostic":"target_profiles","input":request}),
            &[],
            cache,
            || analyze_profiles(request),
            |v| {
                if v["request_digest"]
                    != json!(ContentDigest::sha256(&serde_json::to_vec(request)?))
                {
                    bail!("profile request replay mismatch");
                }
                Ok(())
            },
        )
    }

    #[doc(hidden)]
    pub mod energy_extensions {
        use super::*;
        use rayon::prelude::*;
        use xc_numerics::mpfr_interval::MpfrInterval as M;

        fn zero() -> I {
            I::point(Rational::from(0))
        }
        fn scalar(n: i32) -> I {
            I::point(Rational::from(n))
        }
        fn abs(x: &I) -> I {
            if x.contains_zero() {
                I::new(
                    Rational::from(0),
                    x.lower().clone().abs().max(x.upper().clone().abs()),
                )
                .unwrap()
            } else if x.is_strictly_negative() {
                x.neg()
            } else {
                x.clone()
            }
        }
        fn nonnegative(s: &str) -> Result<Rational> {
            let v = rational(s)?;
            if v < 0 {
                bail!("negative error or norm bound");
            }
            Ok(v)
        }
        fn mp(x: &I, p: u32) -> M {
            use rug::float::Round;
            M::new(
                Float::with_val_round(p, x.lower(), Round::Down).0,
                Float::with_val_round(p, x.upper(), Round::Up).0,
            )
            .unwrap()
        }
        fn qi(x: &M) -> Result<I> {
            Ok(x.try_to_rational_interval()?)
        }
        fn precision(p: u32) -> Result<()> {
            if !(64..=4096).contains(&p) {
                bail!("energy analysis precision limit");
            }
            Ok(())
        }
        fn admission<T: Serialize>(x: &T) -> Result<()> {
            if serde_json::to_vec(x)?.len() as u64
                > crate::ccm::capture_runtime::RESEARCH_INPUT_MAXIMUM_BYTES
            {
                bail!("energy input byte limit");
            }
            xc_core::validate_secret_free(x, "energy extension input")?;
            Ok(())
        }
        #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
        #[serde(deny_unknown_fields)]
        pub struct DeclaredBound {
            pub upper: String,
            pub provenance: String,
            pub scope: String,
        }
        impl DeclaredBound {
            fn value(&self) -> Result<Rational> {
                if self.provenance.trim().is_empty()
                    || self.scope.trim().is_empty()
                    || self.provenance.len() > 16384
                    || self.scope.len() > 16384
                {
                    bail!("declared bound requires provenance and scope");
                }
                nonnegative(&self.upper)
            }
        }
        #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
        #[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
        pub enum FourierRemainder {
            Unknown,
            /// The mathematical object is exactly the supplied polynomial.
            ExactPolynomial,
            Declared {
                l2: DeclaredBound,
                h1: DeclaredBound,
            },
        }
        #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
        #[serde(deny_unknown_fields)]
        pub struct FormContinuity {
            pub operator_id: String,
            pub period: String,
            pub basis_id: String,
            pub domain: String,
            pub bound: DeclaredBound,
        }
        #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
        #[serde(deny_unknown_fields)]
        pub struct ProjectionRequest {
            pub definition_digest: ContentDigest,
            pub operator_id: String,
            pub period: String,
            pub basis_id: String,
            pub domain: String,
            pub coefficients: ComplexVector,
            pub retained_modes: usize,
            pub remainder: FourierRemainder,
            pub source_l2_error: DeclaredBound,
            pub source_h1_error: DeclaredBound,
            pub continuity: Option<FormContinuity>,
            pub trial_absolute_energy: Option<DeclaredBound>,
            pub precision_bits: u32,
        }
        /// Fourier coefficients are orthonormal exp(2*pi*i*j*t/L)/sqrt(L).
        /// H1 is on the declared interval, not the discontinuous zero extension.
        pub fn analyze_projection(r: &ProjectionRequest) -> Result<Value> {
            admission(r)?;
            precision(r.precision_bits)?;
            let n = r.coefficients.real.len();
            let period = rational(&r.period)?;
            if n == 0
                || n % 2 != 1
                || n > 16385
                || r.retained_modes > n / 2
                || period <= 0
                || !r.definition_digest.validate()
                || r.operator_id.trim().is_empty()
                || r.basis_id != "orthonormal_periodic_fourier"
                || r.domain != "interval_h1"
            {
                bail!("projection definition, basis, domain or dimension mismatch");
            }
            if let Some(c) = &r.continuity {
                if c.operator_id != r.operator_id
                    || rational(&c.period)? != period
                    || c.basis_id != r.basis_id
                    || c.domain != r.domain
                {
                    bail!("projection continuity form/domain mismatch");
                }
                c.bound.value()?;
            }
            if let Some(e) = &r.trial_absolute_energy {
                e.value()?;
            }
            let coeffs = complex_coefficients(&r.coefficients, n)?;
            let frequency = qi(&M::pi(r.precision_bits)
                .mul(&mp(&scalar(2), r.precision_bits))
                .div(&mp(&I::point(period.clone()), r.precision_bits))?)?;
            let mut l2 = zero();
            let mut h1 = zero();
            let mut tail_l2 = zero();
            let mut tail_h1 = zero();
            let mut endpoints = (zero(), zero());
            let mut projected_endpoints = (zero(), zero());
            for (j, z) in coeffs.iter().enumerate() {
                let index = j as i32 - (n / 2) as i32;
                let norm = abs_squared(z);
                let weighted = norm.mul(&scalar(1).add(&frequency.mul(&scalar(index)).square()));
                l2 = checked_interval(l2.add(&norm))?;
                h1 = checked_interval(h1.add(&weighted))?;
                if index.unsigned_abs() as usize > r.retained_modes {
                    tail_l2 = checked_interval(tail_l2.add(&norm))?;
                    tail_h1 = checked_interval(tail_h1.add(&weighted))?;
                } else {
                    let sign = scalar(if index % 2 == 0 { 1 } else { -1 });
                    projected_endpoints.0 =
                        checked_interval(projected_endpoints.0.add(&z.0.mul(&sign)))?;
                    projected_endpoints.1 =
                        checked_interval(projected_endpoints.1.add(&z.1.mul(&sign)))?;
                }
                endpoints.0 =
                    endpoints
                        .0
                        .add(&z.0.mul(&scalar(if index % 2 == 0 { 1 } else { -1 })));
                endpoints.1 =
                    endpoints
                        .1
                        .add(&z.1.mul(&scalar(if index % 2 == 0 { 1 } else { -1 })));
            }
            let root_period = sqrt_bound(&I::point(period), r.precision_bits)?;
            endpoints = (
                endpoints.0.div(&root_period)?,
                endpoints.1.div(&root_period)?,
            );
            projected_endpoints = (
                projected_endpoints.0.div(&root_period)?,
                projected_endpoints.1.div(&root_period)?,
            );
            let source_l2 = r.source_l2_error.value()?;
            let source_h1 = r.source_h1_error.value()?;
            let unseen = match &r.remainder {
                FourierRemainder::Unknown => None,
                FourierRemainder::ExactPolynomial => Some((Rational::from(0), Rational::from(0))),
                FourierRemainder::Declared { l2, h1 } => Some((l2.value()?, h1.value()?)),
            };
            let mut result = json!({"status":"unresolved","definition":r,
                "finite_l2_squared":pack_bound(&l2),"finite_h1_squared":pack_bound(&h1),
                "observed_tail_l2_squared":pack_bound(&tail_l2),"observed_tail_h1_squared":pack_bound(&tail_h1),
                "observed_through_modes":n/2,"retained_modes":r.retained_modes,
                "endpoint_traces":pack_complex(&endpoints),
                "projected_endpoint_traces":pack_complex(&projected_endpoints),
                "scope":"interval H1; finite observed tail is not an unseen-tail bound; supplied premises are not proved by this evaluator"});
            let Some((unseen_l2, unseen_h1)) = unseen else {
                result["reason"] = json!("unseen Fourier tail unresolved");
                return Ok(result);
            };
            let eta =
                sqrt_bound(&tail_l2, r.precision_bits)?.upper().clone() + unseen_l2 + &source_l2;
            let rho =
                sqrt_bound(&tail_h1, r.precision_bits)?.upper().clone() + &unseen_h1 + &source_h1;
            let h = sqrt_bound(&h1, r.precision_bits)?.upper().clone() + unseen_h1 + source_h1;
            let m = (sqrt_bound(&l2, r.precision_bits)?.lower().clone() - source_l2)
                .max(Rational::from(0));
            result["l2_projection_error_upper"] = json!(eta.to_string());
            result["h1_projection_error_upper"] = json!(rho.to_string());
            result["trial_h1_upper"] = json!(h.to_string());
            result["trial_l2_floor"] = json!(m.to_string());
            let Some(c) = &r.continuity else {
                result["reason"] = json!("form continuity bound unavailable");
                return Ok(result);
            };
            let b = c.bound.value()?;
            let delta = checked_interval(
                I::point(b).mul(
                    &I::point(h)
                        .mul(&I::point(rho.clone()))
                        .mul(&scalar(2))
                        .add(&I::point(rho).square()),
                ),
            )?;
            result["energy_difference_upper"] = json!(delta.upper().to_string());
            result["status"] = json!("conditional_energy_bound");
            if m <= eta {
                result["quotient_status"] = json!("unresolved_norm_floor");
            } else if let Some(e) = &r.trial_absolute_energy {
                let floor = m - eta;
                let quotient = I::point(e.value()?)
                    .add(&delta)
                    .div(&I::point(floor.clone()).square())?;
                result["projected_norm_floor"] = json!(floor.to_string());
                result["absolute_rayleigh_upper"] = json!(quotient.upper().to_string());
                result["quotient_status"] = json!("conditional_bound");
            } else {
                result["quotient_status"] = json!("unresolved_trial_energy");
            }
            Ok(result)
        }

        #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
        #[serde(deny_unknown_fields)]
        pub struct RankOne {
            pub weight: te::ExactBounds,
            pub vector: Vec<te::ExactBounds>,
        }
        #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
        #[serde(deny_unknown_fields)]
        pub struct ComponentData {
            pub label: String,
            pub signed_weight: String,
            pub diagonal: Vec<te::ExactBounds>,
            pub upper_triangle: Vec<te::ExactBounds>,
            pub rank_one: Vec<RankOne>,
        }
        #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
        #[serde(deny_unknown_fields)]
        pub struct SignedComponent {
            pub content_digest: ContentDigest,
            pub data: ComponentData,
        }
        impl SignedComponent {
            pub fn from_data(data: ComponentData) -> Result<Self> {
                Ok(Self {
                    content_digest: ContentDigest::sha256(&serde_json::to_vec(&data)?),
                    data,
                })
            }
            fn validate(&self, n: usize) -> Result<()> {
                let d = &self.data;
                if self.content_digest != ContentDigest::sha256(&serde_json::to_vec(d)?)
                    || d.label.trim().is_empty()
                    || d.label.len() > 256
                    || d.rank_one.len() > 64
                    || (!d.diagonal.is_empty() && d.diagonal.len() != n)
                    || (!d.upper_triangle.is_empty() && d.upper_triangle.len() != n * (n + 1) / 2)
                    || d.rank_one.iter().any(|r| r.vector.len() != n)
                {
                    bail!("component content authentication or dimension failure");
                }
                rational(&d.signed_weight)?;
                for x in d.diagonal.iter().chain(&d.upper_triangle).chain(
                    d.rank_one
                        .iter()
                        .flat_map(|r| std::iter::once(&r.weight).chain(&r.vector)),
                ) {
                    read_bound(x)?;
                }
                Ok(())
            }
            fn entry(&self, n: usize, i: usize, j: usize) -> Result<I> {
                let d = &self.data;
                let mut v = if i == j && !d.diagonal.is_empty() {
                    read_bound(&d.diagonal[i])?
                } else {
                    zero()
                };
                if !d.upper_triangle.is_empty() {
                    v = v.add(&read_bound(
                        &d.upper_triangle[i * n - i * i.saturating_sub(1) / 2 + j - i],
                    )?);
                }
                for r in &d.rank_one {
                    v = checked_interval(
                        v.add(
                            &read_bound(&r.weight)?
                                .mul(&read_bound(&r.vector[i])?)
                                .mul(&read_bound(&r.vector[j])?),
                        ),
                    )?;
                }
                checked_interval(v.mul(&I::point(rational(&d.signed_weight)?)))
            }
        }
        #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
        #[serde(deny_unknown_fields)]
        pub struct ComponentRequest {
            pub matrix_digest: ContentDigest,
            pub basis_id: String,
            pub components: Vec<SignedComponent>,
            pub assembly_operator_norm_error: Option<DeclaredBound>,
        }
        struct ComponentReader<'a> {
            c: &'a SignedComponent,
            n: usize,
            complex: bool,
            i: usize,
            j: usize,
            pos: u64,
        }
        impl stream::FormReader for ComponentReader<'_> {
            fn next_block(
                &mut self,
                max: usize,
                bytes: usize,
            ) -> std::result::Result<Option<stream::FormBlock>, xc_solver::SolverError>
            {
                let mut entries = vec![];
                let mut used = 0;
                let start = self.pos;
                let n = self.n * if self.complex { 2 } else { 1 };
                while self.i < n && entries.len() < max {
                    let value = if self.complex && self.i < self.n && self.j >= self.n {
                        Ok(zero())
                    } else {
                        self.c.entry(self.n, self.i % self.n, self.j % self.n)
                    }
                    .map_err(|e| xc_solver::SolverError::InvalidConfiguration(e.to_string()))?;
                    let entry = pack_bound(&value);
                    let size = entry.lower.len() + entry.upper.len();
                    if size > bytes {
                        return Err(xc_solver::SolverError::InvalidConfiguration(
                            "component stream scalar budget".into(),
                        ));
                    }
                    if used + size > bytes {
                        break;
                    }
                    entries.push(entry);
                    used += size;
                    self.pos += 1;
                    self.j += 1;
                    if self.j == n {
                        self.i += 1;
                        self.j = self.i;
                    }
                }
                Ok((!entries.is_empty()).then_some(stream::FormBlock { start, entries }))
            }
        }
        #[allow(clippy::too_many_arguments)]
        pub(super) fn analyze_components(
            r: &ComponentRequest,
            m: &RetainedMatrix<'_>,
            problem: &stream::Problem,
            series: &TrialSeries,
            total: &stream::Report,
            functional: Option<&BoundedFunctional>,
            opts: &stream::Options,
        ) -> Result<Value> {
            admission(r)?;
            let n = 2 * m.modes + 1;
            let complex = problem.operator.metadata.dimension == 2 * n;
            if r.matrix_digest != m.manifest.content_digest
                || r.basis_id != BASIS
                || r.components.is_empty()
                || r.components.len() > 16
            {
                bail!("component decomposition operator identity or count mismatch");
            }
            let mut names = std::collections::BTreeSet::new();
            for c in &r.components {
                c.validate(n)?;
                if !names.insert(&c.data.label) {
                    bail!("duplicate component label");
                }
            }
            let entry_work = (n as u128) * (n as u128 + 1) / 2
                * r.components
                    .iter()
                    .map(|c| 1 + 3 * c.data.rank_one.len() as u128)
                    .sum::<u128>();
            if entry_work > opts.maximum_interval_operations as u128 {
                bail!("component closure operation limit before iteration");
            }
            // Closure defects are formed per row on workers, a bounded block of
            // rows at a time, then accumulated serially in (row, column) order.
            let defect_row = |i: usize| -> Result<Vec<I>> {
                (i..n)
                    .map(|j| {
                        let sum = r.components.iter().try_fold(zero(), |s, c| -> Result<I> {
                            checked_interval(s.add(&c.entry(n, i, j)?))
                        })?;
                        let original = I::point(
                            (exact(&m.entries[i * n + j])? + exact(&m.entries[j * n + i])?) / 2,
                        );
                        Ok(sum.sub(&original))
                    })
                    .collect()
            };
            let mut row_sums = vec![Rational::from(0); n];
            let mut exact_closure = true;
            let mut excluded_zero = false;
            for first in (0..n).step_by(16) {
                let rows = (first..n.min(first + 16))
                    .into_par_iter()
                    .map(defect_row)
                    .collect::<Vec<_>>();
                for (i, row) in (first..).zip(rows) {
                    for (j, defect) in (i..n).zip(row?) {
                        exact_closure &= defect.is_point() && defect.lower() == &0;
                        excluded_zero |= !defect.contains_zero();
                        let size = abs(&defect).upper().clone();
                        row_sums[i] += &size;
                        if i != j {
                            row_sums[j] += size;
                        }
                    }
                }
            }
            let norm_upper = row_sums.into_iter().max().unwrap();
            let token = CancellationToken::new();
            // Components stream independently, each serially; reports and the
            // first failure keep component order.
            xc_numerics::mpfr_interval::ensure_uniform_exponent_range()?;
            let reports = r
                .components
                .par_iter()
                .map(|c| -> Result<_> {
                    let mut cp = problem.clone();
                    cp.operator.metadata.source_id =
                        format!("signed-component:{}", c.content_digest.0);
                    let reader = || ComponentReader {
                        c,
                        n,
                        complex,
                        i: 0,
                        j: 0,
                        pos: 0,
                    };
                    // `c.validate(n)` bound its content digest to its data above.
                    let content = ContentDigest::sha256(&serde_json::to_vec(&(
                        "signed-component-reader-v1",
                        &c.content_digest,
                        n,
                        complex,
                    ))?);
                    cp.operator.sha256 =
                        fingerprint_once(&cp.operator.metadata, opts, &content, || {
                            Ok(stream::fingerprint(
                                &cp.operator.metadata,
                                &mut reader(),
                                opts,
                                &token,
                            )?)
                        })?
                        .sha256;
                    cp.baseline.operator_id = cp.operator.metadata.source_id.clone();
                    for v in &mut cp.corrections {
                        v.operator_id = cp.operator.metadata.source_id.clone();
                    }
                    let report = stream::analyze(&cp, opts, &mut reader(), None, &token)?;
                    let fe = functional
                        .map(|f| functional_stage_energy(f, series, n, &report))
                        .transpose()?;
                    Ok((report, fe))
                })
                .collect::<Vec<_>>()
                .into_iter()
                .collect::<Result<Vec<_>>>()?;
            let mut stages = vec![];
            for (i, t) in total.stages.iter().enumerate() {
                let mut sum = zero();
                let mut absolute = zero();
                for (c, _) in &reports {
                    let e = read_bound(&c.stages[i].measurement.energy)?;
                    sum = sum.add(&e);
                    absolute = absolute.add(&abs(&e));
                }
                let closure = sum.sub(&read_bound(&t.measurement.energy)?);
                let cancellation = if abs(&sum).is_strictly_positive() {
                    Some(pack_bound(&absolute.div(&abs(&sum))?))
                } else {
                    None
                };
                let relative_width = if abs(&sum).is_strictly_positive() {
                    Some(pack_bound(&I::point(sum.width()).div(&abs(&sum))?))
                } else {
                    None
                };
                stages.push(json!({"stage":i,"signed_sum":pack_bound(&sum),"absolute_sum":pack_bound(&absolute),
                    "total_minus_component_sum":pack_bound(&closure.neg()),"absolute_width":sum.width().to_string(),
                    "relative_width":relative_width,"cancellation":cancellation,
                    "assembly_energy_error_upper":r.assembly_operator_norm_error.as_ref().map(|b|->Result<String>{
                        Ok((b.value()?*read_bound(&t.measurement.squared_norm)?.upper()).to_string())}).transpose()?}));
            }
            let components=r.components.iter().zip(&reports).map(|(c,(report,fe))|->Result<Value>{
                let widths=report.stages.iter().map(|s|->Result<Value>{let e=read_bound(&s.measurement.energy)?;
                    Ok(json!({"absolute_width":e.width().to_string(),"relative_width":if abs(&e).is_strictly_positive(){Some(pack_bound(&I::point(e.width()).div(&abs(&e))?))}else{None}}))}).collect::<Result<Vec<_>>>()?;
                Ok(json!({"label":c.data.label,"signed_weight":c.data.signed_weight,"content_digest":c.content_digest,"report":report,"functional_energy":fe,"stage_precision":widths}))
            }).collect::<Result<Vec<_>>>()?;
            Ok(
                json!({"status":"finite_stored_enclosures","components":components,
                "stages":stages,"operator_closure":{"status":if exact_closure{"exact_stored_equality"}else if excluded_zero{"refuted"}else{"enclosed_not_proved_equal"},
                    "spectral_norm_upper":norm_upper.to_string(),"method":"full symmetric entrywise residual; maximum absolute row sum; all vectors covered"},
                "assembly_operator_norm_error":r.assembly_operator_norm_error,
                "scope":"signed stored components; arithmetic enclosure does not include source assembly uncertainty; component labels never determine signs"}),
            )
        }

        #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
        #[serde(deny_unknown_fields)]
        pub struct Knot {
            pub coordinate: String,
            pub value: String,
        }
        #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
        #[serde(deny_unknown_fields)]
        pub struct LogProfile {
            pub label: String,
            pub knots: Vec<Knot>,
        }
        #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
        #[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
        pub enum ProfileScope {
            /// The object being measured is the compact interpolant itself.
            ExactInterpolant,
            UnresolvedSource,
            /// Total energy discrepancy, including unobserved exterior tails
            /// and their cross terms, for each of the three named functions.
            DeclaredEnergyErrors {
                profile_a: Box<DeclaredBound>,
                profile_b: Box<DeclaredBound>,
                profile_sum: Box<DeclaredBound>,
            },
        }
        #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
        #[serde(deny_unknown_fields)]
        pub struct AdditiveOverlap {
            /// Nonnegative-coordinate samples; negative coordinates use reflection.
            pub source: LogProfile,
            pub coordinates: Vec<String>,
            pub fourier_terms: usize,
        }
        #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
        #[serde(deny_unknown_fields)]
        pub struct IntegrationWindow {
            pub label: String,
            pub lower: String,
            pub upper: String,
        }
        #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
        #[serde(deny_unknown_fields)]
        pub struct ContinuousRequest {
            pub definition_digest: ContentDigest,
            pub profile_a: LogProfile,
            pub profile_b: LogProfile,
            pub profile_sum: LogProfile,
            pub scope: ProfileScope,
            pub summands: Vec<LogProfile>,
            #[serde(default)]
            pub integration_windows: Vec<IntegrationWindow>,
            #[serde(default)]
            pub unrepresented_sum_l2: Option<DeclaredBound>,
            pub overlap: Option<AdditiveOverlap>,
            pub translation_shifts: Vec<String>,
            pub integration_cells: usize,
            pub maximum_prime_power: usize,
            pub precision_bits: u32,
        }
        #[derive(Clone)]
        struct Pl {
            x: Vec<Rational>,
            y: Vec<Rational>,
        }
        impl Pl {
            fn parse(r: &LogProfile, zero_endpoints: bool) -> Result<Self> {
                if r.label.trim().is_empty()
                    || r.label.len() > 256
                    || !(2..=512).contains(&r.knots.len())
                {
                    bail!("piecewise linear profile identity or knot limit");
                }
                let x = r
                    .knots
                    .iter()
                    .map(|v| rational(&v.coordinate))
                    .collect::<Result<Vec<_>>>()?;
                let y = r
                    .knots
                    .iter()
                    .map(|v| rational(&v.value))
                    .collect::<Result<Vec<_>>>()?;
                if x.windows(2).any(|v| v[0] >= v[1])
                    || (zero_endpoints && (y[0] != 0 || y[y.len() - 1] != 0))
                {
                    bail!("profile must have increasing knots and continuous zero extension");
                }
                if x.iter().any(|v| v.clone().abs() > 1024) {
                    bail!("profile coordinate limit");
                }
                Ok(Self { x, y })
            }
            fn at(&self, t: &Rational) -> Rational {
                if t < &self.x[0] || t > &self.x[self.x.len() - 1] {
                    return Rational::from(0);
                }
                let j = self
                    .x
                    .partition_point(|x| x <= t)
                    .saturating_sub(1)
                    .min(self.x.len() - 2);
                self.y[j].clone()
                    + (&self.y[j + 1] - self.y[j].clone()) * (t.clone() - &self.x[j])
                        / (&self.x[j + 1] - self.x[j].clone())
            }
            fn combine(parts: &[(&Self, i32)]) -> Result<Self> {
                let mut x = parts
                    .iter()
                    .flat_map(|(p, _)| p.x.clone())
                    .collect::<Vec<_>>();
                x.sort();
                x.dedup();
                if x.len() > 4096 {
                    bail!("combined profile knot limit");
                }
                let y = x
                    .iter()
                    .map(|t| {
                        parts
                            .iter()
                            .fold(Rational::from(0), |v, (p, w)| v + p.at(t) * w)
                    })
                    .collect();
                Ok(Self { x, y })
            }
            fn norm2(&self) -> Rational {
                (0..self.x.len() - 1).fold(Rational::from(0), |v, j| {
                    v + (&self.x[j + 1] - self.x[j].clone())
                        * (self.y[j].clone() * &self.y[j]
                            + self.y[j].clone() * &self.y[j + 1]
                            + self.y[j + 1].clone() * &self.y[j + 1])
                        / 3
                })
            }
            fn derivative2(&self) -> Rational {
                (0..self.x.len() - 1).fold(Rational::from(0), |v, j| {
                    let d = self.y[j + 1].clone() - &self.y[j];
                    v + d.clone() * d / (&self.x[j + 1] - self.x[j].clone())
                })
            }
            fn endpoint_atoms(&self) -> Rational {
                self.y[0].clone().abs() + self.y[self.y.len() - 1].clone().abs()
            }
            fn variation(&self) -> Rational {
                self.y.windows(2).fold(self.endpoint_atoms(), |v, y| {
                    v + (y[1].clone() - &y[0]).abs()
                })
            }
            fn correlation(&self, shift: &Rational) -> Rational {
                let left = self.x[0].clone().max(&self.x[0] - shift.clone());
                let right = self.x[self.x.len() - 1]
                    .clone()
                    .min(&self.x[self.x.len() - 1] - shift.clone());
                if left >= right {
                    return Rational::from(0);
                }
                let mut nodes = vec![left.clone(), right.clone()];
                for x in &self.x {
                    for t in [x.clone(), x - shift.clone()] {
                        if t > left && t < right {
                            nodes.push(t);
                        }
                    }
                }
                nodes.sort();
                nodes.dedup();
                nodes.windows(2).fold(Rational::from(0), |sum, x| {
                    let a = self.at(&x[0]);
                    let b = self.at(&x[1]);
                    let c = self.at(&(x[0].clone() + shift));
                    let d = self.at(&(x[1].clone() + shift));
                    sum + (&x[1] - x[0].clone())
                        * (a.clone() * &c * 2 + a * d.clone() + b.clone() * c + b * d * 2)
                        / 6
                })
            }
            fn correlation_interval(&self, s: &I, p: u32) -> Result<I> {
                let span = self.x[self.x.len() - 1].clone() - &self.x[0];
                if s.lower() >= &span || s.upper() <= &(-span.clone()) {
                    return Ok(zero());
                }
                let center = s.midpoint();
                let radius = s.width() / 2;
                let lip = sqrt_bound(&I::point(self.norm2() * self.derivative2()), p)?
                    .upper()
                    .clone()
                    .min(self.derivative2() * s.lower().clone().abs().max(s.upper().clone().abs()));
                let v = self.correlation(&center);
                let e = lip * radius;
                Ok(I::new(v.clone() - &e, v + e)?)
            }
            fn moment(&self, rate: i32, p: u32) -> Result<I> {
                let q = mp(&I::point(Rational::from((rate, 2))), p);
                let mut sum = mp(&zero(), p);
                for j in 0..self.x.len() - 1 {
                    let a =
                        (self.y[j + 1].clone() - &self.y[j]) / (&self.x[j + 1] - self.x[j].clone());
                    let b = self.y[j].clone() - a.clone() * &self.x[j];
                    let antiderivative = |x: &Rational| -> Result<M> {
                        let xx = mp(&I::point(x.clone()), p);
                        Ok(mp(&I::point(a.clone()), p)
                            .mul(&xx)
                            .add(&mp(&I::point(b.clone()), p))
                            .div(&q)?
                            .sub(&mp(&I::point(a.clone()), p).div(&q.square())?)
                            .mul(&q.mul(&xx).exp()))
                    };
                    sum =
                        sum.add(&antiderivative(&self.x[j + 1])?.sub(&antiderivative(&self.x[j])?));
                }
                qi(&sum)
            }
        }
        // Integral of exp(s/2)/sinh(s) from s to infinity; evaluated away from zero.
        fn arch_tail(s: &I, p: u32) -> Result<M> {
            let t = mp(s, p)
                .mul(&mp(&I::point(Rational::from((-1, 2))), p))
                .exp();
            let one = mp(&scalar(1), p);
            Ok(t.atan()
                .mul(&mp(&scalar(2), p))
                .add(&one.add(&t).div(&one.sub(&t))?.ln()?))
        }
        fn arch_weight(s: &I, p: u32) -> Result<M> {
            let v = mp(s, p);
            let sinh = v.exp().sub(&v.neg().exp()).div(&mp(&scalar(2), p))?;
            Ok(v.mul(&mp(&I::point(Rational::from((1, 2))), p))
                .exp()
                .div(&sinh)?)
        }
        fn primes(max: usize) -> Vec<usize> {
            let mut flags = vec![true; max + 1];
            let mut out = vec![];
            for p in 2..=max {
                if flags[p] {
                    out.push(p);
                    if p <= max / p {
                        for j in (p * p..=max).step_by(p) {
                            flags[j] = false;
                        }
                    }
                }
            }
            out
        }
        fn form(pf: &Pl, cells: usize, max_prime: usize, p: u32) -> Result<Value> {
            if pf.endpoint_atoms() != 0 {
                bail!("energy domain requires continuous zero extension; endpoint atoms retained separately");
            }
            let span = pf.x[pf.x.len() - 1].clone() - &pf.x[0];
            let ceiling = qi(&mp(&I::point(span.clone()), p).exp())?
                .upper()
                .clone()
                .ceil()
                .numer()
                .clone();
            let limit = ceiling
                .to_usize()
                .ok_or_else(|| anyhow::anyhow!("prime support exceeds work limit"))?;
            if limit > max_prime {
                bail!("prime support exceeds declared work limit; no truncated prime sum reported as complete");
            }
            let c0 = pf.norm2();
            let d2 = pf.derivative2();
            // Prime-power terms and integration cells are formed on workers and
            // added serially in their original order; the first failure in that
            // order is reported.
            xc_numerics::mpfr_interval::ensure_uniform_exponent_range()?;
            let mut powers = vec![];
            for base in primes(limit) {
                let mut power = base;
                let mut exponent = 1i32;
                while power <= limit {
                    powers.push((base, power, exponent));
                    if power > limit / base {
                        break;
                    }
                    power *= base;
                    exponent += 1;
                }
            }
            let mut prime = mp(&zero(), p);
            let terms = powers.len();
            for term in powers
                .into_par_iter()
                .map(|(base, power, exponent)| -> Result<M> {
                    let log = mp(&I::point(Rational::from(base)), p).ln()?;
                    let shift = qi(&log.mul(&mp(&scalar(exponent), p)))?;
                    let corr = pf.correlation_interval(&shift, p)?;
                    Ok(log
                        .mul(&mp(&corr, p))
                        .mul(&mp(&scalar(2), p))
                        .div(&mp(&I::point(Rational::from(power)), p).sqrt()?)?)
                })
                .collect::<Vec<_>>()
            {
                prime = prime.add(&term?);
            }
            let pole = mp(&pf.moment(1, p)?, p)
                .mul(&mp(&pf.moment(-1, p)?, p))
                .mul(&mp(&scalar(2), p));
            let step = span.clone() / cells;
            // C(s)-C(0)=-||R(.-s)-R||^2/2 and ||T_sR-R|| <= |s| ||R'||.
            // sinh(s)>=s bounds the first cell without subtracting nearly equal numbers.
            let near_upper = qi(&mp(&I::point(d2.clone() * &step * &step / 4), p)
                .mul(&mp(&I::point(step.clone() / 2), p).exp()))?
            .upper()
            .clone();
            let mut integral = mp(&I::new(-near_upper, Rational::from(0))?, p);
            let correlation_lipschitz = sqrt_bound(&I::point(c0.clone() * &d2), p)?.upper().clone();
            let cell = |j: usize| -> Result<M> {
                let a = step.clone() * j;
                let b = step.clone() * (j + 1);
                let s = I::new(a, b.clone())?;
                let raw = pf.correlation_interval(&s, p)?.sub(&I::point(c0.clone()));
                let bound: Rational = d2.clone() * &b * &b / 2;
                let loss = I::new(
                    raw.lower().clone().max(-bound.clone()),
                    raw.upper().clone().min(Rational::from(0)),
                )?;
                let weight = arch_weight(&s, p)?;
                let range = qi(&mp(&loss, p)
                    .mul(&weight)
                    .mul(&mp(&I::point(step.clone()), p)))?;
                // C''(s)=-<R',T_s R'> is continuous, including at knot differences.
                // The midpoint remainder is h^3 sup|((C-C0)w)''|/24.
                let sm = mp(&s, p);
                let positive = sm.exp();
                let negative = sm.neg().exp();
                let sinh = positive.sub(&negative).div(&mp(&scalar(2), p))?;
                let coth = positive.add(&negative).div(&positive.sub(&negative))?;
                let log_derivative = mp(&I::point(Rational::from((1, 2))), p).sub(&coth);
                let first = weight.mul(&log_derivative);
                let second = weight.mul(
                    &log_derivative
                        .square()
                        .add(&mp(&scalar(1), p).div(&sinh.square())?),
                );
                let derivative_bound = correlation_lipschitz.clone().min(d2.clone() * &b);
                let second_bound = I::point(d2.clone())
                    .mul(&abs(&qi(&weight)?))
                    .add(&I::point(derivative_bound * 2i32).mul(&abs(&qi(&first)?)))
                    .add(&I::point(bound.min(c0.clone() * 2i32)).mul(&abs(&qi(&second)?)));
                let error = second_bound
                    .mul(&I::point(step.clone() * &step * &step / 24))
                    .upper()
                    .clone();
                let middle = s.midpoint();
                let midpoint = qi(&mp(&I::point(pf.correlation(&middle) - &c0), p)
                    .mul(&arch_weight(&I::point(middle), p)?)
                    .mul(&mp(&I::point(step.clone()), p)))?
                .add(&I::new(-error.clone(), error)?);
                let enclosure = I::new(
                    range.lower().clone().max(midpoint.lower().clone()),
                    range.upper().clone().min(midpoint.upper().clone()),
                )?;
                Ok(mp(&enclosure, p))
            };
            for value in (1..cells).into_par_iter().map(cell).collect::<Vec<_>>() {
                integral = integral.add(&value?);
            }
            integral =
                integral.sub(&mp(&I::point(c0.clone()), p).mul(&arch_tail(&I::point(span), p)?));
            let pi = M::pi(p);
            let constant = pi
                .mul(&mp(&scalar(8), p))
                .ln()?
                .add(&M::euler_gamma(p))
                .add(&pi.div(&mp(&scalar(2), p))?);
            let arch = mp(&I::point(c0.clone()), p).mul(&constant).add(&integral);
            let energy = qi(&pole.sub(&arch).sub(&prime))?;
            let rayleigh = if c0 > 0 {
                Some(pack_bound(&energy.div(&I::point(c0.clone()))?))
            } else {
                None
            };
            let abs_sum = abs(&qi(&pole)?)
                .add(&abs(&qi(&arch)?))
                .add(&abs(&qi(&prime)?));
            let cancellation = if abs(&energy).is_strictly_positive() {
                Some(pack_bound(&abs_sum.div(&abs(&energy))?))
            } else {
                None
            };
            Ok(
                json!({"energy":pack_bound(&energy),"squared_norm":c0.to_string(),"derivative_squared_norm":d2.to_string(),
                "rayleigh":rayleigh,"pole":pack_bound(&qi(&pole)?),"archimedean":pack_bound(&qi(&arch)?),"prime":pack_bound(&qi(&prime)?),
                "pole_moments":{"positive":pack_bound(&pf.moment(1,p)?),"negative":pack_bound(&pf.moment(-1,p)?)},
                "prime_power_terms":terms,"integration_cells":cells,"archimedean_tail":"analytic integral beyond full correlation support",
                "quadrature":"analytic zero cell; midpoint plus rigorous second-derivative remainder intersected with interval range cells",
                "absolute_width":energy.width().to_string(),"cancellation":cancellation,
                "convention":"real log profile; Q=pole-archimedean-prime; C(s)=integral R(t)R(t+s)dt",
                "scope":"direct whole-line form on the exact compact continuous piecewise-linear interpolant; directed quadrature and prime logarithms; no operator identity is assumed"}),
            )
        }
        fn energy_value(v: &Value) -> Result<I> {
            read_bound(&serde_json::from_value::<te::ExactBounds>(
                v["energy"].clone(),
            )?)
        }

        /// The direct and Fourier sums use independent formulas for the same
        /// even compact interpolant. The Fourier tail follows from TV(f').
        pub fn analyze_overlap(r: &AdditiveOverlap, p: u32) -> Result<Value> {
            admission(r)?;
            precision(p)?;
            let f = Pl::parse(&r.source, false)?;
            if f.x[0] != 0
                || f.y[f.y.len() - 1] != 0
                || r.coordinates.len() > 128
                || r.coordinates.is_empty()
                || !(1..=65536).contains(&r.fourier_terms)
                || r.coordinates.len() * r.fourier_terms * f.x.len() > 4_000_000
            {
                bail!("additive overlap support or work limit");
            }
            let slopes = (0..f.x.len() - 1)
                .map(|j| (f.y[j + 1].clone() - &f.y[j]) / (&f.x[j + 1] - f.x[j].clone()))
                .collect::<Vec<_>>();
            let tv: Rational = 2
                * (slopes[0].clone().abs()
                    + slopes[slopes.len() - 1].clone().abs()
                    + slopes
                        .windows(2)
                        .map(|a| (a[1].clone() - &a[0]).abs())
                        .sum::<Rational>());
            let integral: Rational = 2
                * (0..f.x.len() - 1)
                    .map(|j| (&f.x[j + 1] - f.x[j].clone()) * (&f.y[j + 1] + f.y[j].clone()) / 2)
                    .sum::<Rational>();
            let mut rows = vec![];
            for coordinate in &r.coordinates {
                let u = rational(coordinate)?;
                if u <= 0 {
                    bail!("additive overlap coordinate must be positive");
                }
                let last = (f.x[f.x.len() - 1].clone() / &u)
                    .floor()
                    .numer()
                    .to_usize()
                    .ok_or_else(|| anyhow::anyhow!("direct summand count overflow"))?;
                if last > 1_000_000 {
                    bail!("direct summand work limit");
                }
                let root = mp(&I::point(u.clone()), p).sqrt()?;
                let direct = mp(
                    &I::point((1..=last).map(|n| f.at(&(u.clone() * n))).sum()),
                    p,
                )
                .mul(&root);
                let mut fourier = mp(&zero(), p);
                for n in 1..=r.fourier_terms {
                    let w = M::pi(p).mul(&mp(&I::point(Rational::from(n) * 2 / &u), p));
                    let mut value = mp(&zero(), p);
                    for (j, a) in slopes.iter().enumerate() {
                        let endpoint = |i: usize| -> Result<M> {
                            let z = w.mul(&mp(&I::point(f.x[i].clone()), p));
                            Ok(mp(&I::point(f.y[i].clone()), p)
                                .mul(&z.sin())
                                .div(&w)?
                                .add(&mp(&I::point(a.clone()), p).mul(&z.cos()).div(&w.square())?))
                        };
                        value = value.add(&endpoint(j + 1)?.sub(&endpoint(j)?));
                    }
                    fourier = fourier.add(&value.mul(&mp(&scalar(2), p)));
                }
                let tail = mp(&I::point(tv.clone() * &u * &u), p)
                    .div(
                        &M::pi(p)
                            .square()
                            .mul(&mp(&scalar(4), p))
                            .mul(&mp(&I::point(Rational::from(r.fourier_terms)), p)),
                    )?
                    .div(&root)?;
                let poisson = fourier.div(&root)?;
                let correction = mp(&I::point(integral.clone()), p)
                    .div(&root)?
                    .sub(&root.mul(&mp(&I::point(f.y[0].clone()), p)))
                    .div(&mp(&scalar(2), p))?;
                let trunc = qi(&poisson.add(&correction))?;
                let radius = qi(&tail)?.upper().clone();
                let enclosed = trunc.add(&I::new(-radius.clone(), radius.clone())?);
                let defect = qi(&direct)?.sub(&enclosed);
                rows.push(json!({"u":coordinate,"direct":pack_bound(&qi(&direct)?),"direct_term_count":last,
                    "fourier_partial":pack_bound(&qi(&poisson)?),"pole_correction":pack_bound(&qi(&correction)?),
                    "poisson_enclosure":pack_bound(&enclosed),"direct_minus_poisson":pack_bound(&defect),
                    "overlap_consistent":defect.contains_zero(),"omitted_fourier_terms_upper":radius.to_string(),
                    "error_ledger":{"source":"exact interpolant","quadrature":"analytic segment integrals with directed rounding",
                        "direct_omission":"zero beyond compact support","fourier_omission":"TV(f')/(2*pi*xi)^2 and sum(n>K)1/n^2<=1/K"}}));
            }
            Ok(
                json!({"rows":rows,"source_value_at_zero":f.y[0].to_string(),"source_integral":integral.to_string(),
                "derivative_total_variation":tv.to_string(),"zero_constraints_hold":f.y[0]==0 && integral==0,
                "zero_correction_constraints_hold":f.y[0]==0 && integral==0,
                "scope":"independent direct and Poisson routes for an even compact piecewise-linear source; nonzero additive constraints retain the pole correction"}),
            )
        }

        pub fn analyze_continuous(r: &ContinuousRequest) -> Result<Value> {
            admission(r)?;
            precision(r.precision_bits)?;
            if !r.definition_digest.validate()
                || !(4..=16384).contains(&r.integration_cells)
                || !(2..=1_000_000).contains(&r.maximum_prime_power)
                || r.summands.len() > 128
                || r.translation_shifts.len() > 128
                || r.integration_windows.len() > 32
            {
                bail!("continuous evaluator identity or work limit");
            }
            let a = Pl::parse(&r.profile_a, true)?;
            let b = Pl::parse(&r.profile_b, true)?;
            let sum = Pl::parse(&r.profile_sum, true)?;
            let sum_plus_a = Pl::combine(&[(&sum, 1), (&a, 1)])?;
            let sum_plus_b = Pl::combine(&[(&sum, 1), (&b, 1)])?;
            // Work is admitted before the prime and cell loops, including all cross evaluations.
            let knots = [&a, &b, &sum, &sum_plus_a, &sum_plus_b]
                .iter()
                .map(|f| f.x.len())
                .sum::<usize>();
            if knots.saturating_mul(r.integration_cells + r.maximum_prime_power) > 32_000_000 {
                bail!("continuous evaluator operation limit");
            }
            // The five independent forms run concurrently; failures keep their order.
            let [energy_a, energy_b, energy_sum, energy_sum_plus_a, energy_sum_plus_b] =
                [&a, &b, &sum, &sum_plus_a, &sum_plus_b]
                    .into_par_iter()
                    .map(|f| {
                        form(
                            f,
                            r.integration_cells,
                            r.maximum_prime_power,
                            r.precision_bits,
                        )
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
                    .collect::<Result<Vec<_>>>()?
                    .try_into()
                    .expect("five continuous forms");
            let sum_with_a = energy_value(&energy_sum_plus_a)?
                .sub(&energy_value(&energy_sum)?)
                .sub(&energy_value(&energy_a)?)
                .div(&scalar(2))?;
            let sum_with_b = energy_value(&energy_sum_plus_b)?
                .sub(&energy_value(&energy_sum)?)
                .sub(&energy_value(&energy_b)?)
                .div(&scalar(2))?;
            let defect = energy_value(&energy_a)?.sub(&energy_value(&energy_b)?);
            let decomposition = Pl::combine(&[(&sum, 1), (&a, -1), (&b, -1)])?;
            let mut result = json!({"status":"finite_interpolant_enclosures","definition_digest":r.definition_digest,"scope":r.scope,
                "profile_a":energy_a,"profile_b":energy_b,"profile_sum":energy_sum,"sum_with_a":pack_bound(&sum_with_a),"sum_with_b":pack_bound(&sum_with_b),
                "difference_energy":pack_bound(&defect),"decomposition_l2_squared":decomposition.norm2().to_string(),
                "identity_status":"no_identity_among_supplied_profiles_is_asserted",
                "domains":{"profile_a":[a.x[0].to_string(),a.x[a.x.len()-1].to_string()],"profile_b":[b.x[0].to_string(),b.x[b.x.len()-1].to_string()],
                    "profile_sum":[sum.x[0].to_string(),sum.x[sum.x.len()-1].to_string()]},
                "source_global_status":"unresolved","error_ledger":{"arithmetic_and_quadrature":"enclosed by each energy interval",
                    "prime_sum":"all prime powers within full correlation support","exterior_correlation":"analytic archimedean tail",
                    "source_approximation":"separate declared scope, never inferred from grid refinement"}});
            match &r.scope {
                ProfileScope::ExactInterpolant => {
                    result["source_global_status"] = json!("exact_compact_interpolants_only")
                }
                ProfileScope::UnresolvedSource => {}
                ProfileScope::DeclaredEnergyErrors {
                    profile_a: a,
                    profile_b: b,
                    profile_sum: sum,
                } => {
                    let error_a = a.value()?;
                    let error_b = b.value()?;
                    let error_sum = sum.value()?;
                    let radius = error_a.clone() + &error_b;
                    result["conditional_source_difference_energy"] =
                        json!(pack_bound(&defect.add(&I::new(-radius.clone(), radius)?)));
                    for (name, error) in [
                        ("profile_a", error_a),
                        ("profile_b", error_b),
                        ("profile_sum", error_sum),
                    ] {
                        result[name]["conditional_source_energy"] = json!(pack_bound(
                            &energy_value(&result[name])?.add(&I::new(-error.clone(), error)?)
                        ));
                    }
                    result["source_global_status"] =
                        json!("conditional_on_declared_energy_error_ledgers");
                }
            }
            let mut translations = vec![];
            for shift in &r.translation_shifts {
                let s = rational(shift)?;
                let corr = b.correlation(&s);
                let delta: Rational = (b.norm2() - &corr) * 2;
                // For R supported through b, overlap starts at a-shift and ends at b-shift.
                translations.push(json!({"shift":shift,"correlation":corr.to_string(),"translated_difference_squared_norm":delta.to_string(),
                    "derivative_bound":(s.clone()*s*b.derivative2()).to_string(),
                    "shifted_support":[(&b.x[0]-rational(shift)?).to_string(),(&b.x[b.x.len()-1]-rational(shift)?).to_string()]}));
            }
            result["translations"] = json!(translations);
            let summands = r
                .summands
                .iter()
                .map(|x| Pl::parse(x, false))
                .collect::<Result<Vec<_>>>()?;
            if summands.is_empty()
                && (!r.integration_windows.is_empty() || r.unrepresented_sum_l2.is_some())
            {
                bail!(
                    "summand windows and remainder bounds require the retained partial summand sum"
                );
            }
            if !summands.is_empty() {
                // Hard-cutoff summand pieces have derivative-measure atoms. For their
                // L2 sum, use open-cell endpoint traces rather than bridging jumps.
                let mut nodes = summands
                    .iter()
                    .flat_map(|a| a.x.clone())
                    .collect::<Vec<_>>();
                nodes.sort();
                nodes.dedup();
                if nodes.len() * summands.len() > 262144 {
                    bail!("summand work limit");
                }
                let mut norm = Rational::from(0);
                for cell in nodes.windows(2) {
                    let mid = (&cell[0] + cell[1].clone()) / 2;
                    let mut a = Rational::from(0);
                    let mut b = a.clone();
                    for f in &summands {
                        if mid > f.x[0] && mid < f.x[f.x.len() - 1] {
                            a += f.at(&cell[0]);
                            b += f.at(&cell[1]);
                        }
                    }
                    norm += (&cell[1] - cell[0].clone())
                        * (a.clone() * &a + a * &b + b.clone() * b)
                        / 3;
                }
                let diagonal = summands.iter().map(Pl::norm2).sum::<Rational>();
                let mut windows = vec![];
                for window in &r.integration_windows {
                    let lo = rational(&window.lower)?;
                    let hi = rational(&window.upper)?;
                    if lo >= hi || window.label.trim().is_empty() {
                        bail!("summand window definition");
                    }
                    let norms = summands
                        .iter()
                        .map(|f| {
                            let left = lo.clone().max(f.x[0].clone());
                            let right = hi.clone().min(f.x[f.x.len() - 1].clone());
                            if left >= right {
                                return Rational::from(0);
                            }
                            let mut x = vec![left.clone()];
                            x.extend(f.x.iter().filter(|v| **v > left && **v < right).cloned());
                            x.push(right);
                            let y = x.iter().map(|v| f.at(v)).collect();
                            Pl { x, y }.norm2()
                        })
                        .collect::<Vec<_>>();
                    windows.push(json!({"window":window,"summand_norms_squared":norms.iter().map(ToString::to_string).collect::<Vec<_>>(),
                        "first_summand_squared_norm":norms[0].to_string()}));
                }
                let tail = if let Some(b) = &r.unrepresented_sum_l2 {
                    let epsilon = b.value()?;
                    let partial = sqrt_bound(&I::point(norm.clone()), r.precision_bits)?;
                    Some(
                        json!({"status":"conditional_bound","premise":b,"sum_norm_upper":(partial.upper().clone()+&epsilon).to_string(),
                        "squared_norm_error_upper":(partial.upper().clone()*epsilon.clone()*2i32+epsilon.clone()*epsilon).to_string()}),
                    )
                } else {
                    None
                };
                result["summands"] = json!({"norms_squared":summands.iter().map(|a|a.norm2().to_string()).collect::<Vec<_>>(),
                    "diagonal_sum":diagonal.to_string(),"sum_norm_squared":norm.to_string(),"aggregate_off_diagonal":(norm-diagonal).to_string(),
                    "endpoint_atom_variations":summands.iter().map(|a|a.endpoint_atoms().to_string()).collect::<Vec<_>>(),
                    "derivative_measure_total_variations":summands.iter().map(|a|a.variation().to_string()).collect::<Vec<_>>(),
                    "windows":windows,"unrepresented_sum_bound":tail,"tail_status":if tail.is_some(){"conditional"}else{"unresolved"},
                    "scope":"supplied summands only; no omitted-summand bound inferred; O(total knots) grid and no Gram matrix"});
            }
            result["overlap"] = r
                .overlap
                .as_ref()
                .map(|o| analyze_overlap(o, r.precision_bits))
                .transpose()?
                .unwrap_or(Value::Null);
            Ok(result)
        }

        pub fn capture_projection(
            r: &ProjectionRequest,
            cache: &ArtifactCacheContext<'_>,
        ) -> Result<ArtifactExecutionCacheResult<ResearchRecord<Value>>> {
            managed(
                KIND,
                json!({"semantics":SEMANTICS,"diagnostic":"projection_energy","request":r}),
                &[],
                cache,
                || analyze_projection(r),
                |v| {
                    if v.get("status").is_none() {
                        bail!("projection capture status absent");
                    }
                    Ok(())
                },
            )
        }
        pub fn capture_continuous(
            r: &ContinuousRequest,
            cache: &ArtifactCacheContext<'_>,
        ) -> Result<ArtifactExecutionCacheResult<ResearchRecord<Value>>> {
            managed(
                KIND,
                json!({"semantics":SEMANTICS,"diagnostic":"continuous_energy","request":r}),
                &[],
                cache,
                || analyze_continuous(r),
                |v| {
                    if v.get("status").is_none() {
                        bail!("continuous capture status absent");
                    }
                    Ok(())
                },
            )
        }

        #[cfg(test)]
        mod parallel_form_tests {
            use super::*;

            #[test]
            fn retained_stream_fingerprint_matches_fresh_and_binds_exact_entries() {
                use super::super::{fingerprint_once, matrix_content, MatrixReader};
                let p = 128;
                let entries = (0..25)
                    .map(|k| Float::with_val(p, k as i64 - 7) / 3u32)
                    .collect::<Vec<_>>();
                let bytes = b"fingerprint matrix";
                let digest = ContentDigest::sha256(bytes);
                let manifest: ArtifactManifest = serde_json::from_value(json!({
                    "schema_version":1,"key":xc_cache::ArtifactKey::new("ccm_tau_matrix","fingerprint-test",bytes).unwrap(),
                    "content_digest":digest,"size_bytes":bytes.len(),"objects":[{"content_digest":digest,"size_bytes":bytes.len()}],
                    "created_unix_seconds":1,"producer_toolkit_version":xc_cache::ToolkitVersion::parse("0.16.0").unwrap(),
                    "minimum_reader_version":xc_cache::ToolkitVersion::parse("0.16.0").unwrap(),"maximum_reader_version":null,
                    "quality":"validated","visibility":"private","immutable":true,"dependencies":[],"tags":{},"provenance_digest":null
                }))
                .unwrap();
                let m = RetainedMatrix::from_admitted_runtime(
                    manifest.clone(),
                    "13".into(),
                    2,
                    p,
                    &entries,
                )
                .unwrap();
                let metadata = stream::FormMetadata {
                    source_id: format!(
                        "sym:fingerprint fixture {:?}",
                        std::time::SystemTime::now()
                    ),
                    basis_id: BASIS.into(),
                    normalization_id: NORMALIZATION.into(),
                    dimension: 5,
                };
                let opts = stream::Options::default();
                let token = CancellationToken::new();
                let fresh =
                    stream::fingerprint(&metadata, &mut MatrixReader::new(&m), &opts, &token)
                        .unwrap();
                let content = matrix_content(&m, false);
                let first = fingerprint_once(&metadata, &opts, &content, || {
                    Ok(stream::fingerprint(
                        &metadata,
                        &mut MatrixReader::new(&m),
                        &opts,
                        &token,
                    )?)
                })
                .unwrap();
                let retained = fingerprint_once(&metadata, &opts, &content, || {
                    panic!("an identical fingerprint was recomputed")
                })
                .unwrap();
                assert_eq!(first, fresh);
                assert_eq!(retained, fresh);
                // Any exact entry change, or the complex reading, changes the key.
                let mut changed = entries.clone();
                changed[7].next_up();
                let other =
                    RetainedMatrix::from_admitted_runtime(manifest, "13".into(), 2, p, &changed)
                        .unwrap();
                assert_ne!(matrix_content(&other, false), content);
                assert_ne!(matrix_content(&m, true), content);
            }

            fn profile(knots: &[(&str, &str)]) -> Pl {
                let profile = LogProfile {
                    label: "parallel form fixture".into(),
                    knots: knots
                        .iter()
                        .map(|(x, y)| Knot {
                            coordinate: (*x).into(),
                            value: (*y).into(),
                        })
                        .collect(),
                };
                Pl::parse(&profile, true).unwrap()
            }

            #[test]
            fn parallel_form_is_bit_identical_to_serial_reference_at_any_thread_count() {
                let profiles = [
                    profile(&[("-1/4", "0"), ("0", "1"), ("1/4", "0")]),
                    profile(&[("-3/2", "0"), ("-1/3", "2/7"), ("1/5", "-1"), ("2", "0")]),
                    profile(&[("0", "0"), ("1/2", "3"), ("1", "0")]),
                ];
                for pf in &profiles {
                    for (cells, max_prime, p) in [(64, 8, 128), (257, 1024, 320)] {
                        let expected = reference_form(pf, cells, max_prime, p);
                        for threads in [1, 2, 4, 8] {
                            let pool = rayon::ThreadPoolBuilder::new()
                                .num_threads(threads)
                                .build()
                                .unwrap();
                            for _ in 0..2 {
                                let actual = pool.install(|| form(pf, cells, max_prime, p));
                                match (&actual, &expected) {
                                    (Ok(a), Ok(b)) => assert_eq!(a, b),
                                    (Err(a), Err(b)) => assert_eq!(a.to_string(), b.to_string()),
                                    _ => panic!("parallel and serial forms disagree on failure"),
                                }
                            }
                        }
                    }
                }
            }

            /// Serial `form` retained verbatim from before parallel terms.
            fn reference_form(pf: &Pl, cells: usize, max_prime: usize, p: u32) -> Result<Value> {
                if pf.endpoint_atoms() != 0 {
                    bail!("energy domain requires continuous zero extension; endpoint atoms retained separately");
                }
                let span = pf.x[pf.x.len() - 1].clone() - &pf.x[0];
                let ceiling = qi(&mp(&I::point(span.clone()), p).exp())?
                    .upper()
                    .clone()
                    .ceil()
                    .numer()
                    .clone();
                let limit = ceiling
                    .to_usize()
                    .ok_or_else(|| anyhow::anyhow!("prime support exceeds work limit"))?;
                if limit > max_prime {
                    bail!("prime support exceeds declared work limit; no truncated prime sum reported as complete");
                }
                let c0 = pf.norm2();
                let d2 = pf.derivative2();
                let mut prime = mp(&zero(), p);
                let mut terms = 0usize;
                for base in primes(limit) {
                    let log = mp(&I::point(Rational::from(base)), p).ln()?;
                    let mut power = base;
                    let mut exponent = 1i32;
                    while power <= limit {
                        let shift = qi(&log.mul(&mp(&scalar(exponent), p)))?;
                        let corr = pf.correlation_interval(&shift, p)?;
                        prime = prime.add(
                            &log.mul(&mp(&corr, p))
                                .mul(&mp(&scalar(2), p))
                                .div(&mp(&I::point(Rational::from(power)), p).sqrt()?)?,
                        );
                        terms += 1;
                        if power > limit / base {
                            break;
                        }
                        power *= base;
                        exponent += 1;
                    }
                }
                let pole = mp(&pf.moment(1, p)?, p)
                    .mul(&mp(&pf.moment(-1, p)?, p))
                    .mul(&mp(&scalar(2), p));
                let step = span.clone() / cells;
                // C(s)-C(0)=-||R(.-s)-R||^2/2 and ||T_sR-R|| <= |s| ||R'||.
                // sinh(s)>=s bounds the first cell without subtracting nearly equal numbers.
                let near_upper = qi(&mp(&I::point(d2.clone() * &step * &step / 4), p)
                    .mul(&mp(&I::point(step.clone() / 2), p).exp()))?
                .upper()
                .clone();
                let mut integral = mp(&I::new(-near_upper, Rational::from(0))?, p);
                let correlation_lipschitz =
                    sqrt_bound(&I::point(c0.clone() * &d2), p)?.upper().clone();
                for j in 1..cells {
                    let a = step.clone() * j;
                    let b = step.clone() * (j + 1);
                    let s = I::new(a, b.clone())?;
                    let raw = pf.correlation_interval(&s, p)?.sub(&I::point(c0.clone()));
                    let bound: Rational = d2.clone() * &b * &b / 2;
                    let loss = I::new(
                        raw.lower().clone().max(-bound.clone()),
                        raw.upper().clone().min(Rational::from(0)),
                    )?;
                    let weight = arch_weight(&s, p)?;
                    let range = qi(&mp(&loss, p)
                        .mul(&weight)
                        .mul(&mp(&I::point(step.clone()), p)))?;
                    // C''(s)=-<R',T_s R'> is continuous, including at knot differences.
                    // The midpoint remainder is h^3 sup|((C-C0)w)''|/24.
                    let sm = mp(&s, p);
                    let positive = sm.exp();
                    let negative = sm.neg().exp();
                    let sinh = positive.sub(&negative).div(&mp(&scalar(2), p))?;
                    let coth = positive.add(&negative).div(&positive.sub(&negative))?;
                    let log_derivative = mp(&I::point(Rational::from((1, 2))), p).sub(&coth);
                    let first = weight.mul(&log_derivative);
                    let second = weight.mul(
                        &log_derivative
                            .square()
                            .add(&mp(&scalar(1), p).div(&sinh.square())?),
                    );
                    let derivative_bound = correlation_lipschitz.clone().min(d2.clone() * &b);
                    let second_bound = I::point(d2.clone())
                        .mul(&abs(&qi(&weight)?))
                        .add(&I::point(derivative_bound * 2i32).mul(&abs(&qi(&first)?)))
                        .add(&I::point(bound.min(c0.clone() * 2i32)).mul(&abs(&qi(&second)?)));
                    let error = second_bound
                        .mul(&I::point(step.clone() * &step * &step / 24))
                        .upper()
                        .clone();
                    let middle = s.midpoint();
                    let midpoint = qi(&mp(&I::point(pf.correlation(&middle) - &c0), p)
                        .mul(&arch_weight(&I::point(middle), p)?)
                        .mul(&mp(&I::point(step.clone()), p)))?
                    .add(&I::new(-error.clone(), error)?);
                    let enclosure = I::new(
                        range.lower().clone().max(midpoint.lower().clone()),
                        range.upper().clone().min(midpoint.upper().clone()),
                    )?;
                    integral = integral.add(&mp(&enclosure, p));
                }
                integral = integral
                    .sub(&mp(&I::point(c0.clone()), p).mul(&arch_tail(&I::point(span), p)?));
                let pi = M::pi(p);
                let constant = pi
                    .mul(&mp(&scalar(8), p))
                    .ln()?
                    .add(&M::euler_gamma(p))
                    .add(&pi.div(&mp(&scalar(2), p))?);
                let arch = mp(&I::point(c0.clone()), p).mul(&constant).add(&integral);
                let energy = qi(&pole.sub(&arch).sub(&prime))?;
                let rayleigh = if c0 > 0 {
                    Some(pack_bound(&energy.div(&I::point(c0.clone()))?))
                } else {
                    None
                };
                let abs_sum = abs(&qi(&pole)?)
                    .add(&abs(&qi(&arch)?))
                    .add(&abs(&qi(&prime)?));
                let cancellation = if abs(&energy).is_strictly_positive() {
                    Some(pack_bound(&abs_sum.div(&abs(&energy))?))
                } else {
                    None
                };
                Ok(
                    json!({"energy":pack_bound(&energy),"squared_norm":c0.to_string(),"derivative_squared_norm":d2.to_string(),
                    "rayleigh":rayleigh,"pole":pack_bound(&qi(&pole)?),"archimedean":pack_bound(&qi(&arch)?),"prime":pack_bound(&qi(&prime)?),
                    "pole_moments":{"positive":pack_bound(&pf.moment(1,p)?),"negative":pack_bound(&pf.moment(-1,p)?)},
                    "prime_power_terms":terms,"integration_cells":cells,"archimedean_tail":"analytic integral beyond full correlation support",
                    "quadrature":"analytic zero cell; midpoint plus rigorous second-derivative remainder intersected with interval range cells",
                    "absolute_width":energy.width().to_string(),"cancellation":cancellation,
                    "convention":"real log profile; Q=pole-archimedean-prime; C(s)=integral R(t)R(t+s)dt",
                    "scope":"direct whole-line form on the exact compact continuous piecewise-linear interpolant; directed quadrature and prime logarithms; no operator identity is assumed"}),
                )
            }
        }
    }

    // Complete upper triangle of the symmetric part. This preserves exactly
    // x^T Tau x even for historical stored matrices with tiny asymmetry.
    struct MatrixReader<'a> {
        matrix: &'a RetainedMatrix<'a>,
        complex: bool,
        i: usize,
        j: usize,
        position: u64,
    }
    impl<'a> MatrixReader<'a> {
        fn new(matrix: &'a RetainedMatrix<'a>) -> Self {
            Self {
                matrix,
                complex: false,
                i: 0,
                j: 0,
                position: 0,
            }
        }
        fn for_complex(matrix: &'a RetainedMatrix<'a>, complex: bool) -> Self {
            Self {
                complex,
                ..Self::new(matrix)
            }
        }
    }
    impl stream::FormReader for MatrixReader<'_> {
        fn next_block(
            &mut self,
            maximum_entries: usize,
            maximum_scalar_bytes: usize,
        ) -> std::result::Result<Option<stream::FormBlock>, xc_solver::SolverError> {
            let invalid = |s: String| xc_solver::SolverError::InvalidConfiguration(s);
            let physical = 2 * self.matrix.modes + 1;
            let n = physical * if self.complex { 2 } else { 1 };
            let start = self.position;
            let mut entries = Vec::new();
            let mut bytes = 0;
            while self.i < n && entries.len() < maximum_entries {
                if self.complex && self.i < physical && self.j >= physical {
                    if bytes + 2 > maximum_scalar_bytes {
                        break;
                    }
                    entries.push(te::ExactBounds::point("0"));
                    bytes += 2;
                    self.position += 1;
                    self.j += 1;
                    if self.j == n {
                        self.i += 1;
                        self.j = self.i;
                    }
                    continue;
                }
                let i = self.i % physical;
                let j = self.j % physical;
                let left = &self.matrix.entries[i * physical + j];
                let right = &self.matrix.entries[j * physical + i];
                let (an, ad) = float_bits(left).map_err(|e| invalid(e.to_string()))?;
                let (bn, bd) = float_bits(right).map_err(|e| invalid(e.to_string()))?;
                // Dyadic addition/halving needs at most these decimal bytes.
                // Admit before allocating a potentially enormous rational/string.
                let estimated =
                    2 * (((an.max(bn) + ad.max(bd) + 4) * 30103).div_ceil(100000) + 4) as usize;
                if estimated > maximum_scalar_bytes {
                    return Err(invalid(
                        "matrix scalar exceeds block budget before conversion".into(),
                    ));
                }
                if bytes + estimated > maximum_scalar_bytes {
                    break;
                }
                let a = exact(left).map_err(|e| invalid(e.to_string()))?;
                let b = exact(right).map_err(|e| invalid(e.to_string()))?;
                let entry = point((a + b) / 2);
                let size = entry.lower.len() + entry.upper.len();
                if size > maximum_scalar_bytes {
                    return Err(invalid("matrix scalar exceeds block budget".into()));
                }
                if bytes + size > maximum_scalar_bytes {
                    break;
                }
                bytes += size;
                entries.push(entry);
                self.position += 1;
                self.j += 1;
                if self.j == n {
                    self.i += 1;
                    self.j = self.i;
                }
            }
            Ok((!entries.is_empty()).then_some(stream::FormBlock { start, entries }))
        }
    }
    fn energy(
        s: &RetainedState,
        m: &RetainedMatrix<'_>,
        input: Option<&ExternalResearchInputs>,
        parity: bool,
    ) -> Result<(Value, bool)> {
        let extra = input.and_then(|i| i.finite_diagnostics.as_ref());
        let opts = extra.map_or_else(stream::Options::default, |i| i.streaming_options.clone());
        let token = CancellationToken::new();
        let physical = s.coefficients.len();
        let complex = extra.and_then(|i| i.complex_trials.as_ref());
        let n = physical * if complex.is_some() { 2 } else { 1 };
        let flatten = |v: &ComplexVector| -> Result<Vector> {
            if v.real.len() != physical || v.imaginary.len() != physical {
                bail!("complex trial coefficients do not match retained matrix");
            }
            Ok(Vector {
                label: v.label.clone(),
                coefficients: v.real.iter().chain(&v.imaginary).cloned().collect(),
            })
        };
        let mut series = if let Some(t) = complex {
            if t.matrix_digest != m.manifest.content_digest {
                bail!("complex trial matrix digest mismatch");
            }
            TrialSeries {
                basis_id: "realification_re_im_of_centered_full_V_fourier".into(),
                scope: t.scope.clone(),
                baseline: flatten(&t.baseline)?,
                corrections: t.corrections.iter().map(flatten).collect::<Result<_>>()?,
                provenance: t.provenance.clone(),
            }
        } else if let Some(t) = extra.and_then(|i| i.trials.as_ref()) {
            t.clone()
        } else {
            let coefficients = s
                .coefficients
                .iter()
                .map(|x| exact(x).map(point))
                .collect::<Result<Vec<_>>>()?;
            TrialSeries {basis_id:BASIS.into(),scope:"raw retained source coefficients; optional sampled finite target projection; projection and source errors excluded".into(),
                baseline:Vector{label:"retained_state".into(),coefficients},corrections:vec![],provenance:BTreeMap::new()}
        };
        if complex.is_none() && extra.and_then(|i| i.trials.as_ref()).is_none() {
            let normalized = match center_normalized(s, input) {
                Ok(value) => value,
                Err(e)
                    if matches!(
                        e.downcast_ref::<xc_solver::SolverError>(),
                        Some(xc_solver::SolverError::PremiseNotVerified(_))
                    ) =>
                {
                    series
                        .provenance
                        .insert("automatic_target_correction_withheld".into(), e.to_string());
                    None
                }
                Err(e) => return Err(e),
            };
            if let Some((x, y, a, b)) = normalized {
                series.baseline = Vector {
                    label: "center_normalized_retained_state".into(),
                    coefficients: x.iter().cloned().map(point).collect(),
                };
                series.corrections.push(Vector {
                    label: "center_normalized_target_minus_state".into(),
                    coefficients: y
                        .iter()
                        .zip(&x)
                        .map(|(y, x)| point(Rational::from(y - x)))
                        .collect(),
                });
                series.scope = "both stored coefficient vectors divided by their exact center functional sum_j (-1)^j v_j; finite target projection; source and projection errors excluded".into();
                series
                    .provenance
                    .insert("source_center_normalizer".into(), a.to_string());
                series
                    .provenance
                    .insert("target_center_normalizer".into(), b.to_string());
            }
        }
        for v in std::iter::once(&series.baseline).chain(&series.corrections) {
            if v.coefficients.len() != n || v.label.trim().is_empty() || v.label.len() > 256 {
                bail!("trial coefficients do not match retained full matrix");
            }
            for bound in &v.coefficients {
                read_bound(bound)?;
            }
        }
        let original = series.clone();
        if parity {
            use xc_numerics::interval::RationalInterval as I;
            let read = |x: &te::ExactBounds| -> Result<I> {
                // Reuse the exact literal reader, including finite decimal input.
                Ok(I::new(rational(&x.lower)?, rational(&x.upper)?)?)
            };
            let mut full = series
                .baseline
                .coefficients
                .iter()
                .map(read)
                .collect::<Result<Vec<_>>>()?;
            for v in &series.corrections {
                for (x, y) in full.iter_mut().zip(&v.coefficients) {
                    *x = x.add(&read(y)?);
                }
            }
            let half = I::point(Rational::from((1, 2)));
            let pack = |x: I| te::ExactBounds {
                lower: x.lower().to_string(),
                upper: x.upper().to_string(),
            };
            let reflected: Vec<_> = full
                .chunks(physical)
                .flat_map(|chunk| chunk.iter().rev())
                .collect();
            let even = full
                .iter()
                .zip(&reflected)
                .map(|(a, b)| pack(a.add(b).mul(&half)))
                .collect();
            let odd = full
                .iter()
                .zip(&reflected)
                .map(|(a, b)| pack(a.sub(b).mul(&half)))
                .collect();
            series.baseline = Vector {
                label: "reciprocal_even_projection".into(),
                coefficients: even,
            };
            series.corrections = vec![Vector {
                label: "reciprocal_odd_projection".into(),
                coefficients: odd,
            }];
        }
        let metadata = stream::FormMetadata {
            source_id: format!(
                "{}:{}",
                if complex.is_some() {
                    "realification_diag_sym_sym"
                } else {
                    "sym"
                },
                m.manifest.content_digest.0
            ),
            basis_id: series.basis_id.clone(),
            normalization_id: NORMALIZATION.into(),
            dimension: n,
        };
        let metric = stream::FormMetadata {
            source_id: "identity-coefficient-metric-v1".into(),
            ..metadata.clone()
        };
        let digest = fingerprint_once(
            &metadata,
            &opts,
            &matrix_content(m, complex.is_some()),
            || {
                Ok(stream::fingerprint(
                    &metadata,
                    &mut MatrixReader::for_complex(m, complex.is_some()),
                    &opts,
                    &token,
                )?)
            },
        )?;
        let vector = |v: &Vector| te::TrialVector {
            label: v.label.clone(),
            source_id: ContentDigest::sha256(&serde_json::to_vec(v).unwrap()).0,
            operator_id: metadata.source_id.clone(),
            metric_id: metric.source_id.clone(),
            basis_id: metadata.basis_id.clone(),
            normalization_id: NORMALIZATION.into(),
            coefficients: v.coefficients.clone(),
        };
        let problem = stream::Problem {
            schema_version: 1,
            operator: stream::FormSource {
                metadata: metadata.clone(),
                sha256: digest.sha256,
            },
            metric: stream::Metric::Identity {
                metadata: metric.clone(),
            },
            baseline: vector(&series.baseline),
            corrections: series.corrections.iter().map(vector).collect(),
        };
        let report = stream::analyze(
            &problem,
            &opts,
            &mut MatrixReader::for_complex(m, complex.is_some()),
            None,
            &token,
        )?;
        let automatic_functional = automatic_profile_request(s, input)?.map(|r| r.functional);
        let functional_energy = complex
            .and_then(|t| t.functional.as_ref())
            .or(automatic_functional.as_ref())
            .map(|f| functional_stage_energy(f, &series, physical, &report))
            .transpose()?;
        let component_energy = extra
            .and_then(|i| i.component_energy.as_ref())
            .map(|r| {
                energy_extensions::analyze_components(
                    r,
                    m,
                    &problem,
                    &series,
                    &report,
                    complex
                        .and_then(|t| t.functional.as_ref())
                        .or(automatic_functional.as_ref()),
                    &opts,
                )
            })
            .transpose()?;
        let energy_distance = extra.and_then(|i|i.energy_distance.as_ref()).map(|premises| -> Result<Value> {
            if premises.matrix_digest!=m.manifest.content_digest {bail!("energy-distance matrix digest mismatch");}
            match report.stages.last().and_then(|stage|stage.measurement.normalized.as_ref()) {
                Some(measurement)=>energy_distance_bound(premises,&measurement.rayleigh_quotient.upper,opts.norm_precision_bits),
                None=>Ok(json!({"status":"unresolved","reason":"final norm is not separated from zero"})),
            }
        }).transpose()?;
        let certified = report.stages.last().is_some_and(|stage| {
            stage.measurement.status == stream::MeasurementStatus::CertifiedFinite
        });
        let parity_summary = if parity {
            let full = &report.stages.last().unwrap().measurement;
            let even = &report.parts[0];
            let odd = &report.parts[1];
            let ratio =
                |a: &te::ExactBounds, b: &te::ExactBounds| -> Result<Option<te::ExactBounds>> {
                    let b = read_bound(b)?;
                    if b.contains_zero() {
                        Ok(None)
                    } else {
                        Ok(Some(pack_bound(&read_bound(a)?.div(&b)?)))
                    }
                };
            Some(
                json!({"odd_squared_norm_fraction":ratio(&odd.squared_norm,&full.squared_norm)?,"odd_signed_energy_fraction":ratio(&odd.energy,&full.energy)?,
                "even_minus_full_energy":pack_bound(&read_bound(&even.energy)?.sub(&read_bound(&full.energy)?)),
                "even_minus_full_rayleigh":match (&even.normalized,&full.normalized) {(Some(e),Some(f))=>Some(pack_bound(&read_bound(&e.rayleigh_quotient)?.sub(&read_bound(&f.rayleigh_quotient)?))),_=>None},
                "energy_cross_term":pack_bound(&read_bound(&report.energy_pairings[1])?.mul(&I::point(Rational::from(2)))),
                "metric_cross_term":pack_bound(&read_bound(&report.metric_pairings[1])?.mul(&I::point(Rational::from(2)))),
                "scope":"measured for this vector; zero cross terms do not establish global parity invariance or positive odd spectrum; small odd norm never removes odd energy"}),
            )
        } else {
            None
        };
        Ok((
            json!({"report":report,"trial_series":original,"complex_trial_series":complex,"effective_problem":problem,"options":opts,
            "coefficient_representation":if complex.is_some() {"complex; realification [Re(z),Im(z)]; diag(sym(Tau),sym(Tau))"} else {"real"},
            "hermitian_scope":"real symmetric stored operator only; complex Hermitian matrices with imaginary entries are not accepted by this adapter; pairings are Re(z* A w)",
            "functional_energy":functional_energy,
            "component_energy":component_energy.unwrap_or_else(||json!({"status":"awaiting_source","reason":"no authenticated signed operator decomposition supplied"})),
            "energy_distance":energy_distance,
            "parity_summary":parity_summary,
            "parity_projection":parity,"reflection":"j -> -j in centered full Fourier coordinates",
            "parity_scope":"decomposition of the final supplied/automatic vector, not a comparison of sector eigenvalues; an exactly even final vector has identically zero odd component",
            "form_scope":"stored symmetric Tau and coefficient identity metric; no assembly/projection/continuum error included; no parity ordering assertion"}),
            certified,
        ))
    }
    fn convergence_result(
        id: &str,
        s: &RetainedState,
        p: &cv::Problem,
        o: &cv::Options,
    ) -> Result<Analysis> {
        let report = match cv::analyze(p, o, &CancellationToken::new()) {
            Ok(report) => report,
            Err(xc_solver::SolverError::PremiseNotVerified(reason)) => {
                let mut a = computed(id, s, json!({"reason":reason}), "premise_not_verified");
                a.reason = Some(reason);
                return Ok(a);
            }
            Err(e) => return Err(e.into()),
        };
        let outcome = match &report.output {
            cv::Output::Root(r) if r.status != cv::RootStatus::CertifiedLocal => "unresolved",
            cv::Output::Budget(_) => "computed_not_certified",
            _ => "computed_not_certified",
        };
        // Caller-supplied bounds and matrices are authenticated as inputs, not
        // equated to the retained CCM operator merely because labels match.
        let mut a = computed(id, s, serde_json::to_value(report)?, outcome);
        a.scope="bounded finite verification of explicitly supplied problem; analytic/source premises not independently established; source labels do not establish equality to CCM".into();
        Ok(a)
    }

    // Every coordinate below is constructed from the authenticated source.
    // Reciprocal-even sources use the orthonormal even compression E^T sym(Tau) E;
    // other sources retain the full space, ordered by increasing |frequency|.
    // Neither construction asserts an infinite-dimensional gap.
    fn automatic_matrix(
        id: &str,
        s: &RetainedState,
        m: &RetainedMatrix<'_>,
        roots: Option<&RetainedRoots>,
    ) -> Result<Analysis> {
        use xc_numerics::mpfr_interval::MpfrInterval as M;
        let full = s.coefficients.len();
        let even = s.coefficients.iter().eq(s.coefficients.iter().rev());
        let d = if even { s.modes + 1 } else { full };
        let p = s.precision.saturating_add(64).max(384);
        if d > 512 || p > 8192 {
            return Ok(empty(id,s,"blocked","automatic finite matrix analysis exceeds dimension 512 or precision 8192; primary sources retained for supplemental analysis"));
        }
        // Workspace admission is the caller's declared working-byte budget,
        // whose dense estimate exceeds this serialized enclosure. The solver's
        // input cap is a fixed constant, so its report never depends on the
        // budget; an input it still rejects is a resource limit, not retained.
        let encode = |x: &M| -> Result<te::ExactBounds> {
            x.validate()?;
            Ok(te::ExactBounds {
                lower: exact(x.lower())?.to_string(),
                upper: exact(x.upper())?.to_string(),
            })
        };
        let scale = s
            .coefficients
            .iter()
            .map(exact)
            .collect::<Result<Vec<_>>>()?;
        let maximum = scale
            .iter()
            .map(|x| x.clone().abs())
            .max()
            .unwrap_or_default();
        if maximum == 0 {
            bail!("zero retained eigenvector");
        }
        let v: Vec<Rational> = scale.iter().map(|x| Rational::from(x / &maximum)).collect();
        let mu = exact(&Float::with_val(s.precision, Float::parse(&s.eigenvalue)?))?;
        let positive = mu.clone().abs();
        if positive == 0 {
            return Ok(computed(
                id,
                s,
                json!({"reason":"stored eigenvalue is zero; automatic isolation proposal unresolved"}),
                "unresolved",
            ));
        }
        let one = M::from_i64(1, p);
        let pair = one.div(&M::from_i64(2, p).sqrt()?)?;
        let mut basis: Vec<Vec<(usize, M)>> = vec![vec![(s.modes, one.clone())]];
        for j in 1..=s.modes {
            if even {
                basis.push(vec![
                    (s.modes - j, pair.clone()),
                    (s.modes + j, pair.clone()),
                ]);
            } else {
                basis.push(vec![(s.modes - j, one.clone())]);
                basis.push(vec![(s.modes + j, one.clone())]);
            }
        }
        let raw = m.entries.iter().map(exact).collect::<Result<Vec<_>>>()?;
        let a: Vec<Rational> = (0..full * full)
            .map(|k| Rational::from(&raw[k] + &raw[(k % full) * full + k / full]) / 2)
            .collect();
        let compress = |x: &[Rational]| -> Vec<M> {
            basis
                .iter()
                .map(|b| {
                    b.iter().fold(M::from_i64(0, p), |sum, (j, w)| {
                        sum.add(&M::from_rational(&x[*j], p).mul(w))
                    })
                })
                .collect()
        };
        let column = compress(&v);
        let mut entries = vec![te::ExactBounds::point("0"); d * d];
        for i in 0..d {
            for j in 0..=i {
                let mut entry = M::from_i64(0, p);
                for (u, x) in &basis[i] {
                    for (v, y) in &basis[j] {
                        entry = entry.add(&M::from_rational(&a[u * full + v], p).mul(x).mul(y));
                    }
                }
                let entry = encode(&entry)?;
                entries[i * d + j] = entry.clone();
                entries[j * d + i] = entry;
            }
        }
        let basis_id = if even {
            "orthonormal_reciprocal_even_frequency_order"
        } else {
            "full_center_out_frequency_order"
        };
        let form = cv::TrialForm {
            source_id: m.manifest.content_digest.0.clone(),
            basis_id: basis_id.into(),
            normalization_id: "euclidean_orthonormal_coordinates".into(),
            dimension: d,
            entries,
        };
        let options = cv::Options {
            maximum_dimension: 512,
            maximum_input_bytes: cv::MAXIMUM_INPUT_BYTES,
            maximum_interval_operations: 1_000_000_000,
            maximum_rational_bits: 65536,
            working_precision_bits: p,
            ..Default::default()
        };
        let mut detail = json!({"preparation":"automatic_after_primary_v1","matrix":m.manifest.content_digest,
            "eigenpair":s.manifest.content_digest,"basis":basis_id,"full_dimension":full,"analyzed_dimension":d,
            "working_precision_bits":p,"normalization":"stored coefficients divided by exact maximum absolute coefficient",
            "analytic_obligations":["stored-to-analytic assembly error","infinite-mode truncation","cutoff-to-target convergence"]});
        let problem = match id {
            "spectral_cluster_bound" => {
                let radius = Rational::from(&positive * 2);
                let norm = column
                    .iter()
                    .fold(M::from_i64(0, p), |sum, x| sum.add(&x.square()));
                norm.validate()?;
                if norm.lower() <= &0 {
                    return Ok(computed(
                        id,
                        s,
                        json!({"reason":"trial Gram lower bound unresolved"}),
                        "unresolved",
                    ));
                }
                cv::Problem::Cluster(cv::ClusterProblem {
                    matrix: form,
                    spectral_window: te::ExactBounds {
                        lower: Rational::from(&mu - &radius).to_string(),
                        upper: Rational::from(&mu + &radius).to_string(),
                    },
                    columns: vec![column.iter().map(&encode).collect::<Result<_>>()?],
                    shifts: vec![mu.to_string()],
                    gram_lower: (exact(norm.lower())? / 2i32).to_string(),
                    previous_columns: None,
                })
            }
            "finite_tail_bound" => {
                let retained = if even {
                    s.modes / 2 + 1
                } else {
                    2 * (s.modes / 2) + 1
                };
                if retained >= d {
                    return Ok(empty(
                        id,
                        s,
                        "not_applicable",
                        "no omitted finite modes in the declared half-frequency partition",
                    ));
                }
                let mut gershgorin: Option<Rational> = None;
                for i in retained..d {
                    let mut row = rational(&form.entries[i * d + i].lower)? - &mu;
                    for j in retained..d {
                        if i != j {
                            row -= rational(&form.entries[i * d + j].lower)?
                                .abs()
                                .max(rational(&form.entries[i * d + j].upper)?.abs());
                        }
                    }
                    gershgorin = Some(gershgorin.map_or(row.clone(), |old| old.min(row)));
                }
                let gap = gershgorin
                    .filter(|x| x > &0)
                    .map(|x| x / 2)
                    .unwrap_or_else(|| Rational::from(&positive / 8));
                detail["retained_frequency_max"] = json!(s.modes / 2);
                detail["gap_proposal"] = json!(gap.to_string());
                cv::Problem::TailBlock(cv::TailBlockProblem {
                    matrix: form,
                    retained_dimension: retained,
                    shift: point(mu.clone()),
                    gap_lower: gap.to_string(),
                })
            }
            "directional_error_bound" => {
                if mu <= 0 {
                    return Ok(computed(
                        id,
                        s,
                        json!({"reason":"positive stored eigenvalue required for automatic inverse-action proposal"}),
                        "unresolved",
                    ));
                }
                // Compute the defect from exact stored dyadics before enclosing it.
                // Quantizing A first would bury a tiny source residual in cancellation.
                let residual: Vec<Rational> =
                    a.chunks(full)
                        .zip(&v)
                        .map(|(row, x)| {
                            Rational::from(&mu * x)
                                - row.iter().zip(&v).fold(Rational::from(0), |sum, (a, b)| {
                                    sum + Rational::from(a * b)
                                })
                        })
                        .collect();
                let rhs = compress(&residual)
                    .iter()
                    .map(&encode)
                    .collect::<Result<Vec<_>>>()?;
                let mut unit = vec![te::ExactBounds::point("0"); d];
                unit[0] = te::ExactBounds::point("1");
                detail["correction_definition"]=json!("x=A_sector^-1 (mu*v_sector-A_sector*v_sector); one finite inverse-iteration correction, not eigenvector or root displacement");
                detail["gap_proposal"] = json!(Rational::from(&mu / 2).to_string());
                cv::Problem::Directional(cv::DirectionalProblem {
                    matrix: form,
                    rhs,
                    approximate_solution: vec![te::ExactBounds::point("0"); d],
                    functional: unit,
                    approximate_dual: vec![te::ExactBounds::point("0"); d],
                    coercivity_lower: Rational::from(&mu / 2).to_string(),
                })
            }
            _ => bail!("unknown automatic matrix diagnostic"),
        };
        let report = match cv::analyze(&problem, &options, &CancellationToken::new()) {
            Ok(r) => r,
            Err(xc_solver::SolverError::PremiseNotVerified(reason)) => {
                detail["reason"] = json!(reason);
                // Failure to prove a proposed window/gap is an inconclusive result,
                // not a counterexample to a manuscript or a successful certificate.
                return Ok(computed(id, s, detail, "unresolved"));
            }
            Err(xc_solver::SolverError::InvalidConfiguration(reason))
                if reason == "convergence: input byte limit" =>
            {
                return Err(ResourceLimited(format!(
                    "automatic matrix problem exceeds the solver input limit: {reason}"
                ))
                .into());
            }
            Err(e) => return Err(e.into()),
        };
        detail["report"] = serde_json::to_value(&report)?;
        let mut out = computed(id, s, detail, "certified_finite_enclosure");
        out.scope="automatic source-bound finite symmetric stored-form analysis in the declared orthonormal sector; no analytic assembly or continuum guarantee".into();
        if let cv::Output::Directional(r) = &report.output {
            // With zero primal/dual and unit functional, the same computed
            // ||residual||/gamma also bounds the full inverse correction norm.
            let bound = M::from_rational(&rational(&r.correction_error_upper)?, p);
            out.result["inverse_correction_l2_upper"] = json!(r.correction_error_upper);
            out.rows.clear();
            out.rows.push(Row {
                label: "finite_inverse_correction_norm".into(),
                outcome: "certified_finite_enclosure".into(),
                result: json!({"upper":r.correction_error_upper}),
            });
            if let Some(roots) = roots {
                let spacing = M::pi(p)
                    .mul(&M::from_i64(2, p))
                    .div(&M::from_rational(&rational(&s.cutoff)?, p).ln()?)?;
                for root in &roots.dataset.points {
                    let Some(value) = &root.value else {
                        out.rows.push(Row {
                            label: root.ordinal.to_string(),
                            outcome: "unresolved".into(),
                            result: json!({"source_status":root.source_status}),
                        });
                        continue;
                    };
                    let t = M::from_rational(
                        &exact(&Float::with_val(
                            roots.dataset.precision_bits,
                            Float::parse(value)?,
                        ))?,
                        p,
                    );
                    let mut coefficients = Vec::with_capacity(full);
                    for j in 0..full {
                        let den = t.sub(&spacing.mul(&M::from_i64(j as i64 - s.modes as i64, p)));
                        if den.contains_zero() {
                            coefficients.clear();
                            break;
                        }
                        coefficients.push(one.div(&den)?);
                    }
                    if coefficients.is_empty() {
                        out.rows.push(Row {
                            label: root.ordinal.to_string(),
                            outcome: "unresolved".into(),
                            result: json!({"reason":"functional touches a secular pole"}),
                        });
                        continue;
                    }
                    let norm = basis
                        .iter()
                        .map(|b| {
                            b.iter().fold(M::from_i64(0, p), |sum, (j, w)| {
                                sum.add(&coefficients[*j].mul(w))
                            })
                        })
                        .fold(M::from_i64(0, p), |sum, x| sum.add(&x.square()));
                    let error = norm.sqrt()?.mul(&bound);
                    error.validate()?;
                    out.rows.push(Row{label:root.ordinal.to_string(),outcome:"certified_finite_enclosure".into(),result:json!({
                        "root_label":root.ordinal,"source_status":root.source_status,"global_ordinal_certified":false,"functional":"sum x_j/(t-pole_j) for the finite inverse correction",
                        "absolute_upper":exact(error.upper())?.to_string(),"root_displacement_bound":null})});
                }
            }
        }
        Ok(out)
    }
    fn automatic_profile(
        id: &str,
        s: &RetainedState,
        input: Option<&ExternalResearchInputs>,
    ) -> Result<Analysis> {
        use rug::float::Round;
        use xc_numerics::mpfr_interval::MpfrInterval as M;
        let Some((x, y, _, _)) = center_normalized(s, input)? else {
            return Ok(empty(id,s,"awaiting_source","requires the run's finite target projection, prepared after the primary state; no user-supplied numerical bounds required"));
        };
        if !x.iter().eq(x.iter().rev()) || !y.iter().eq(y.iter().rev()) {
            return Ok(empty(id,s,"not_applicable","automatic real profile comparison requires exactly reciprocal-even source and target coefficients; no projection of a natural complex profile is substituted"));
        }
        let p = 512;
        let intervals = (8 * s.modes).clamp(256, 8192);
        let encode = |v: &M| -> Result<te::ExactBounds> {
            v.validate()?;
            Ok(te::ExactBounds {
                lower: exact(v.lower())?.to_string(),
                upper: exact(v.upper())?.to_string(),
            })
        };
        let profile = |v: &[Rational], t: &Rational| -> Result<M> {
            let angle = M::pi(p).mul(&M::from_rational(&(t + Rational::from(1)), p));
            let cosine = angle.cos();
            let mut previous = M::from_i64(1, p);
            let mut current = cosine.clone();
            let mut value = M::from_rational(&v[s.modes], p);
            for k in 1..=s.modes {
                value = value.add(
                    &M::from_rational(&Rational::from(&v[s.modes - k] + &v[s.modes + k]), p)
                        .mul(&current),
                );
                let next = cosine.mul(&current).mul(&M::from_i64(2, p)).sub(&previous);
                previous = current;
                current = next;
            }
            value.validate()?;
            Ok(value)
        };
        let log_c = M::from_rational(&rational(&s.cutoff)?, p).ln()?;
        let weight = |t: &M| -> Result<M> {
            Ok(log_c
                .div(&M::from_i64(2, p))?
                .mul(&log_c.mul(t).div(&M::from_i64(4, p))?.exp()))
        };
        let sample = |v: M| -> Result<String> {
            v.validate()?;
            let mid = Float::with_val(96, Float::with_val(p, v.lower()) + v.upper()) / 2i32;
            Ok(exact(&mid)?.to_string())
        };
        let mut lp = l1::WeightedL1Problem {
            schema_version: 1,
            basis_id: "one_dimensional_span_of_retained_finite_target_projection".into(),
            target_id: s.manifest.content_digest.0.clone(),
            normalization_id: "source_center_one; free_reference_amplitude".into(),
            quadrature_id: "uniform_log_coordinate_64_positive_trapezoid_stored_96bit_points"
                .into(),
            basis: vec![],
            target: vec![],
            weights: vec![],
            constraints: vec![],
            rhs: vec![],
        };
        for j in 0..=64 {
            let t = Rational::from((j, 64));
            lp.basis.push(vec![sample(profile(&y, &t)?)?]);
            lp.target.push(sample(profile(&x, &t)?)?);
            lp.weights
                .push(sample(weight(&M::from_rational(&t, p))?.div(
                    &M::from_i64(if j == 0 || j == 64 { 128 } else { 64 }, p),
                )?)?);
        }
        let fit = l1::fit_weighted_l1(
            &lp,
            &l1::WeightedL1Options::default(),
            &CancellationToken::new(),
        )?;
        if id == "constrained_l1_fit" {
            let mut out = computed(
                id,
                s,
                json!({"problem":lp,"fit":fit,
                "recipe":"reference_amplitude_baseline_v1",
                "basis_dimension":1,"equality_constraints":0,
                "limitation":"discrete amplitude-fit baseline only; no added prolate modes, mean-zero trial-class assertion, or continuous optimality certificate",
                "target_definition":input.and_then(|i|i.target.as_ref()).map(|t|&t.definition_digest)}),
                "certified_finite_enclosure",
            );
            out.scope="exact finite weighted L1 optimization of stored 96-bit samples in the one-dimensional span of the retained finite reference projection; custom constrained trial spaces require an explicit recipe".into();
            return Ok(out);
        }
        let fitted = rational(&fit.witness.coefficients[0])?;
        let mut out = empty(id, s, "computed", "");
        out.reason = None;
        out.scope="directed weighted continuous L1 upper bounds between real stored finite Fourier functions on 1<=u<=sqrt(C); excludes target projection and infinite-reference errors; no continuous optimality assertion".into();
        let options = cv::Options {
            maximum_dimension: 512,
            working_precision_bits: p,
            maximum_rational_bits: 65536,
            ..Default::default()
        };
        for (label, amplitude) in [
            ("center_normalized_reference", Rational::from(1)),
            ("sample_fitted_reference_amplitude", fitted),
        ] {
            let delta: Vec<Rational> = x
                .iter()
                .zip(&y)
                .map(|(x, y)| x - Rational::from(y * &amplitude))
                .collect();
            let curvature = delta
                .iter()
                .enumerate()
                .fold(Rational::from(0), |sum, (j, x)| {
                    sum + x.clone().abs() * j.abs_diff(s.modes).pow(2)
                });
            let curvature = M::pi(p).square().mul(&M::from_rational(&curvature, p));
            let curvature = encode(&curvature)?.upper;
            let mut cells = Vec::with_capacity(intervals);
            let mut left = Rational::from(0);
            let mut left_value = encode(&profile(&delta, &left)?)?;
            for j in 0..intervals {
                let right = Rational::from((j + 1, intervals));
                let right_value = encode(&profile(&delta, &right)?)?;
                let t = M::new(
                    Float::with_val_round(p, &left, Round::Down).0,
                    Float::with_val_round(p, &right, Round::Up).0,
                )?;
                cells.push(cv::ProfileCell {
                    left: left.to_string(),
                    right: right.to_string(),
                    left_residual: left_value,
                    right_residual: right_value.clone(),
                    weight: encode(&weight(&t)?)?,
                    second_derivative_upper: curvature.clone(),
                    additional_sup_error_upper: "0".into(),
                });
                left = right;
                left_value = right_value;
            }
            let problem = cv::Problem::Profile(cv::ProfileProblem {
                source_id: format!("{}:{label}", s.manifest.content_digest.0),
                validation_grid_id: format!(
                    "uniform_log_coordinate_{intervals}_directed_curvature"
                ),
                training_grid_id: Some(lp.quadrature_id.clone()),
                cells,
            });
            let report = cv::analyze(&problem, &options, &CancellationToken::new())?;
            out.rows.push(Row {
                label: label.into(),
                outcome: "certified_finite_enclosure".into(),
                result: json!({"amplitude":amplitude.to_string(),"report":report}),
            });
        }
        out.result = json!({"coordinate":"t=2 log(u)/log(C)","weight":"(log(C)/2) exp(t log(C)/4)","validation_intervals":intervals,
            "fit_problem_sha256":fit.problem_sha256,"analytic_obligations":["finite projection to original target","finite target approximation to analytic reference"]});
        Ok(out)
    }
    fn compute(
        id: &str,
        s: &RetainedState,
        m: Option<&RetainedMatrix<'_>>,
        roots: Option<&RetainedRoots>,
        input: Option<&ExternalResearchInputs>,
        certificate: Option<&super::super::sector_gap_certificate::PortableCcmSectorGapCertificate>,
    ) -> Result<Analysis> {
        let extra = input.and_then(|i| i.finite_diagnostics.as_ref());
        if let Some(reason) = extra.and_then(|i| i.not_applicable.get(id)) {
            return Ok(empty(id, s, "not_applicable", reason));
        }
        if let Some(p) = extra.and_then(|i| i.convergence.get(id)) {
            return convergence_result(id, s, p, &extra.unwrap().convergence_options);
        }
        match id {
            "constrained_l1_fit" => {
                let Some(p) = extra.and_then(|i| i.l1.as_ref()) else {
                    return automatic_profile(id, s, input);
                };
                let o = &extra.unwrap().l1_options;
                let report = l1::fit_weighted_l1(p, o, &CancellationToken::new())?;
                Ok(computed(
                    id,
                    s,
                    serde_json::to_value(report)?,
                    "certified_finite_enclosure",
                ))
            }
            "trial_vector_energy" | "trial_vector_parity" => {
                let Some(m) = m else {
                    return Ok(empty(
                        id,
                        s,
                        "missing_input",
                        "retained source matrix unavailable",
                    ));
                };
                let (value, certified) = energy(s, m, input, id == "trial_vector_parity")?;
                let mut a = computed(
                    id,
                    s,
                    value,
                    if certified {
                        "certified_finite_enclosure"
                    } else {
                        "unresolved"
                    },
                );
                if let Some(status) = a.result["energy_distance"]["status"].as_str() {
                    a.rows.push(Row {
                        label: "energy_distance".into(),
                        outcome: status.into(),
                        result: a.result["energy_distance"].clone(),
                    });
                }
                a.scope="exact finite stored-form diagnostics; actual staged vectors require explicit trial inputs; no analytic trial or smoothing is synthesized".into();
                Ok(a)
            }
            "finite_root_budget" => root_result(id, s, roots, certificate),
            "indexed_prolate_comparison" => {
                let Ok(c) = s.cutoff.parse::<u64>() else {
                    return Ok(empty(
                        id,
                        s,
                        "not_applicable",
                        "indexed asymptotic API currently requires an integer cutoff",
                    ));
                };
                let mut a=empty(id,s,"computed","numerical indexed prolate deficits and independent Weil ordinal comparisons require explicit comparison inputs; asymptotic points are not certified deficits");
                a.result = prolate_convention();
                use super::super::evidence::{prolate_log_deficiency_asymptotic, EvenProlateMode};
                for index in [4, 6, 8] {
                    let mode = EvenProlateMode::new(index)?;
                    let log = prolate_log_deficiency_asymptotic(mode, c, s.precision)?;
                    a.rows.push(Row{label:format!("full_prolate_index_{index}"),outcome:"computed_not_certified".into(),result:json!({"full_index":index,"descending_even_index":mode.descending_even_index(),"fourier_sign":mode.fourier_sign(),"log_singular_value_deficit_asymptotic":xc_numerics::prefix::lossless_decimal(&log)})});
                }
                for reference in extra
                    .map(|i| i.prolate_references.as_slice())
                    .unwrap_or(&[])
                {
                    let p = s.precision.max(reference.precision_bits);
                    let predicted = prolate_log_deficiency_asymptotic(
                        EvenProlateMode::new(reference.full_index)?,
                        c,
                        p,
                    )?;
                    let numerical =
                        Float::with_val(p, rational(&reference.log_singular_value_deficit)?);
                    // D=-log(deficit), hence D_num-D_asym = log_asym-log_num.
                    let difference = Float::with_val(p, &predicted - &numerical);
                    let transfer = reference
                        .weil
                        .as_ref()
                        .map(|w| -> Result<_> {
                            let mut delta =
                                Float::with_val(p, rational(&w.log_absolute_eigenvalue)?);
                            delta -= &numerical;
                            Ok(xc_numerics::prefix::lossless_decimal(&delta))
                        })
                        .transpose()?;
                    a.rows.push(Row{label:format!("reference_{}_q{}",reference.full_index,reference.quadrature_order),outcome:"computed_not_certified".into(),result:json!({"reference":reference,"D_numerical_minus_D_asymptotic":xc_numerics::prefix::lossless_decimal(&difference),"log_abs_weil_over_numerical_deficit":transfer,"working_precision_bits":p})});
                }
                Ok(a)
            }
            "directional_error_bound" | "finite_tail_bound" | "spectral_cluster_bound" => {
                let Some(m) = m else {
                    return Ok(empty(
                        id,
                        s,
                        "awaiting_source",
                        "retained matrix required after primary acquisition",
                    ));
                };
                automatic_matrix(id, s, m, roots)
            }
            "dimension_precision_budget" => {
                if let Some(c) = extra.and_then(|i| i.refinement_cohort.as_ref()) {
                    if !c.observations.iter().any(|o| {
                        o.source == s.manifest.content_digest
                            && o.configuration.n_modes == s.modes
                            && o.configuration.precision_bits == s.precision
                            && rational(&o.configuration.lambda_squared).ok()
                                == rational(&s.cutoff).ok()
                    }) {
                        bail!("attached refinement cohort must include this retained source at its actual configuration");
                    }
                    Ok(computed(
                        id,
                        s,
                        analyze_refinement_cohort(c)?,
                        "computed_not_certified",
                    ))
                } else {
                    Ok(empty(id,s,"awaiting_cohort","requires independently computed compatible N/P runs; analyze their retained per-root results after acquisition; analytic truncation and cutoff bounds remain separate obligations"))
                }
            }
            "normalization_error_bound" => normalization_result(id, s, input),
            "continuous_l1_bound" => paired_profile_result(id, s, input),
            _ => bail!("unknown finite diagnostic"),
        }
    }
    type CenterNormalized = (Vec<Rational>, Vec<Rational>, Rational, Rational);
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum RefinementAxis {
        Dimension,
        OperatorQuadrature,
        ProjectionQuadrature,
        Representation,
        ArithmeticPrecision,
        GuardBits,
        CertificatePrecision,
    }
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum SpectralBranch {
        EvenGround,
        EvenFirstExcited,
        OddGround,
    }
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum IndexOrigin {
        Zero,
        One,
    }
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum RefinementObservable {
        Eigenvalue,
        RawEnergy,
        SquaredNorm,
        RayleighQuotient,
        FunctionalEnergy,
        CenterProfileDistance,
        FunctionalProfileDistance,
    }
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum ComparisonDomain {
        FullWindow,
        PositiveHalf,
        NegativeHalf,
        CoefficientSpace,
    }
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct RefinementConfiguration {
        pub observable: RefinementObservable,
        pub domain: ComparisonDomain,
        pub lambda_squared: String,
        pub branch: SpectralBranch,
        pub external_index: usize,
        pub index_origin: IndexOrigin,
        pub n_modes: usize,
        pub precision_bits: u32,
        pub operator_quadrature_orders: Vec<usize>,
        pub projection_quadrature_order: usize,
        pub representation_order: usize,
        pub guard_bits: u32,
        pub certificate_precision_bits: u32,
        pub basis_id: String,
        pub metric_id: String,
        pub operator_recipe: ContentDigest,
        pub target_definition: Option<ContentDigest>,
        pub normalizer_id: String,
        pub trial_recipe: Option<ContentDigest>,
    }
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum ObservationValidity {
        Accepted,
        Unresolved,
        Rejected,
    }
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct RefinementObservation {
        pub source: ContentDigest,
        pub matrix: ContentDigest,
        pub configuration: RefinementConfiguration,
        pub validity: ObservationValidity,
        pub value: String,
        pub enclosure: Option<te::ExactBounds>,
    }
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct RefinementCohort {
        pub axis: RefinementAxis,
        pub scope: String,
        pub relative_tolerance: String,
        pub observations: Vec<RefinementObservation>,
    }
    fn canonical_configuration(c: &RefinementConfiguration) -> Result<RefinementConfiguration> {
        let index = match c.branch {
            SpectralBranch::EvenFirstExcited => 1,
            _ => 0,
        };
        let origin = if c.index_origin == IndexOrigin::One {
            1
        } else {
            0
        };
        if c.external_index != index + origin
            || c.n_modes == 0
            || c.n_modes > 8192
            || !(64..=1_000_000).contains(&c.precision_bits)
            || c.operator_quadrature_orders.is_empty()
            || c.operator_quadrature_orders.len() > 8193
            || c.operator_quadrature_orders
                .iter()
                .any(|&q| q == 0 || q > 1_000_000)
            || c.basis_id.trim().is_empty()
            || c.metric_id.trim().is_empty()
            || c.normalizer_id.trim().is_empty()
            || !c.operator_recipe.validate()
            || c.target_definition.as_ref().is_some_and(|d| !d.validate())
            || c.trial_recipe.as_ref().is_some_and(|d| !d.validate())
        {
            bail!("refinement branch, index, configuration or identity invalid");
        }
        let cutoff = rational(&c.lambda_squared)?;
        if cutoff <= 1 {
            bail!("refinement cutoff must exceed one");
        }
        let mut normalized = c.clone();
        normalized.lambda_squared = cutoff.to_string();
        normalized.external_index = index;
        normalized.index_origin = IndexOrigin::Zero;
        // Uniform fixed-Q tables remain one policy across dimensions.
        if normalized
            .operator_quadrature_orders
            .iter()
            .all(|&q| q == normalized.operator_quadrature_orders[0])
        {
            normalized.operator_quadrature_orders.truncate(1);
        }
        Ok(normalized)
    }
    fn without_axis(c: &RefinementConfiguration, axis: RefinementAxis) -> RefinementConfiguration {
        let mut c = c.clone();
        match axis {
            RefinementAxis::Dimension => c.n_modes = 0,
            RefinementAxis::OperatorQuadrature => c.operator_quadrature_orders.clear(),
            RefinementAxis::ProjectionQuadrature => c.projection_quadrature_order = 0,
            RefinementAxis::Representation => c.representation_order = 0,
            RefinementAxis::ArithmeticPrecision => c.precision_bits = 0,
            RefinementAxis::GuardBits => c.guard_bits = 0,
            RefinementAxis::CertificatePrecision => c.certificate_precision_bits = 0,
        }
        c
    }
    fn refinement_increases(
        a: &RefinementConfiguration,
        b: &RefinementConfiguration,
        axis: RefinementAxis,
    ) -> bool {
        match axis {
            RefinementAxis::Dimension => b.n_modes > a.n_modes,
            RefinementAxis::OperatorQuadrature => {
                a.operator_quadrature_orders.len() == b.operator_quadrature_orders.len()
                    && a.operator_quadrature_orders
                        .iter()
                        .zip(&b.operator_quadrature_orders)
                        .all(|(a, b)| b >= a)
                    && a.operator_quadrature_orders != b.operator_quadrature_orders
            }
            RefinementAxis::ProjectionQuadrature => {
                b.projection_quadrature_order > a.projection_quadrature_order
            }
            RefinementAxis::Representation => b.representation_order > a.representation_order,
            RefinementAxis::ArithmeticPrecision => b.precision_bits > a.precision_bits,
            RefinementAxis::GuardBits => b.guard_bits > a.guard_bits,
            RefinementAxis::CertificatePrecision => {
                b.certificate_precision_bits > a.certificate_precision_bits
            }
        }
    }
    #[doc(hidden)]
    pub fn analyze_refinement_cohort(c: &RefinementCohort) -> Result<Value> {
        if c.scope.trim().is_empty()
            || c.scope.len() > 16384
            || !(1..=4096).contains(&c.observations.len())
            || serde_json::to_vec(c)?.len() > 64 * 1024 * 1024
        {
            bail!("refinement cohort scope or work limit");
        }
        let tolerance = rational(&c.relative_tolerance)?;
        if tolerance <= 0 || tolerance >= 1 {
            bail!("refinement tolerance must be between zero and one");
        }
        let configs = c
            .observations
            .iter()
            .map(|o| canonical_configuration(&o.configuration))
            .collect::<Result<Vec<_>>>()?;
        let fixed = without_axis(&configs[0], c.axis);
        let mut sources = std::collections::BTreeSet::new();
        let mut configurations = std::collections::BTreeSet::new();
        let mut np = std::collections::BTreeSet::new();
        let mut rows = vec![];
        let mut consecutive = 0;
        let mut enclosed = 0;
        for (j, (o, config)) in c.observations.iter().zip(&configs).enumerate() {
            if !o.source.validate() || !o.matrix.validate() || without_axis(config, c.axis) != fixed
            {
                bail!("incompatible refinement join: more than the declared axis differs");
            }
            let value = rational(&o.value)?;
            let bounds = o.enclosure.as_ref().map(read_bound).transpose()?;
            if bounds.as_ref().is_some_and(|b| !b.contains(&value)) {
                bail!("point lies outside supplied refinement enclosure");
            }
            let unique = sources.insert(o.source.0.clone())
                && configurations.insert(serde_json::to_vec(config)?);
            np.insert((config.n_modes, config.precision_bits));
            let mut row = json!({"source":o.source,"matrix":o.matrix,"configuration":config,"validity":o.validity,"status":"first_observation"});
            if j > 0 {
                let previous = &c.observations[j - 1];
                row["predecessor"] = json!(previous.source);
                let unchanged_operator = matches!(
                    c.axis,
                    RefinementAxis::ProjectionQuadrature
                        | RefinementAxis::Representation
                        | RefinementAxis::CertificatePrecision
                );
                if unchanged_operator && previous.matrix != o.matrix {
                    bail!("same-operator refinement requires the identical matrix digest");
                }
                // Invalid observations and duplicate replays break the chain.
                // They cannot be skipped to manufacture consecutive changes.
                if unique
                    && refinement_increases(&configs[j - 1], config, c.axis)
                    && rows.last().is_some_and(|r: &Value| r["eligible"] == true)
                    && o.validity == ObservationValidity::Accepted
                    && value != 0
                {
                    let old = rational(&previous.value)?;
                    let change = (&value - old).abs() / value.clone().abs();
                    let passes = change < tolerance;
                    consecutive = if passes { consecutive + 1 } else { 0 };
                    row["status"] = json!(if passes {
                        "change_below_tolerance"
                    } else {
                        "change_not_below_tolerance"
                    });
                    row["relative_change"] = json!(change.to_string());
                    let interval = match (bounds.as_ref(), previous.enclosure.as_ref()) {
                        (Some(new), Some(old)) if !new.contains_zero() => {
                            let d = new.sub(&read_bound(old)?);
                            let magnitude = |v: &I| {
                                I::new(
                                    if v.contains_zero() {
                                        Rational::from(0)
                                    } else {
                                        v.lower().clone().abs().min(v.upper().clone().abs())
                                    },
                                    v.lower().clone().abs().max(v.upper().clone().abs()),
                                )
                            };
                            Some(magnitude(&d)?.div(&magnitude(new)?)?)
                        }
                        _ => None,
                    };
                    let pass_interval = interval.as_ref().is_some_and(|v| v.upper() < &tolerance);
                    enclosed = if pass_interval { enclosed + 1 } else { 0 };
                    row["relative_change_enclosure"] = interval
                        .as_ref()
                        .map_or(Value::Null, |v| json!(pack_bound(v)));
                } else {
                    consecutive = 0;
                    enclosed = 0;
                    row["status"] = json!("invalid_duplicate_or_unresolved_step");
                }
            }
            row["eligible"] = json!(unique && o.validity == ObservationValidity::Accepted);
            row["consecutive_point_changes_below_tolerance"] = json!(consecutive);
            row["consecutive_enclosed_changes_below_tolerance"] = json!(enclosed);
            rows.push(row);
        }
        Ok(
            json!({"semantics":"typed-refinement-cohort-v1","scope":"operational stabilization of supplied finite observations; no error-to-limit estimate or source certificate; interval assertions conditional on supplied enclosures",
            "axis":c.axis,"formula":"abs(new-old)/abs(new); strict comparison before rounding; final two adjacent eligible steps",
            "operationally_stabilized":consecutive>=2,"enclosed_changes_below_tolerance":enclosed>=2,
            "unique_sources":sources.len(),"distinct_configurations":configurations.len(),"distinct_dimension_precision_configurations":np.len(),"rows":rows}),
        )
    }
    #[doc(hidden)]
    pub fn capture_refinement_cohort(
        c: &RefinementCohort,
        cache: &ArtifactCacheContext<'_>,
    ) -> Result<ArtifactExecutionCacheResult<ResearchRecord<Value>>> {
        if cache.write_visibility == xc_cache::CacheVisibility::Public {
            bail!("refinement cohort capture is private only");
        }
        managed(
            KIND,
            json!({"semantics":SEMANTICS,"diagnostic":"refinement_cohort","input":c}),
            &[],
            cache,
            || analyze_refinement_cohort(c),
            |v| {
                if v["semantics"] != "typed-refinement-cohort-v1" {
                    bail!("refinement result semantics mismatch");
                }
                Ok(())
            },
        )
    }
    #[doc(hidden)]
    pub fn analyze_matched_spectral_triple(
        observations: &[RefinementObservation; 3],
    ) -> Result<Value> {
        let mut configurations = observations
            .iter()
            .map(|o| canonical_configuration(&o.configuration))
            .collect::<Result<Vec<_>>>()?;
        let branches = [
            SpectralBranch::EvenGround,
            SpectralBranch::OddGround,
            SpectralBranch::EvenFirstExcited,
        ];
        let mut values = Vec::new();
        for (o, branch) in observations.iter().zip(branches) {
            if o.configuration.observable != RefinementObservable::Eigenvalue
                || o.configuration.branch != branch
                || !o.source.validate()
                || !o.matrix.validate()
                || o.matrix != observations[0].matrix
                || o.validity != ObservationValidity::Accepted
            {
                bail!("matched triple requires accepted even-ground, odd-ground, even-first-excited observations from the same matrix");
            }
            let v = rational(&o.value)?;
            let enclosure = o.enclosure.as_ref().map(read_bound).transpose()?;
            if enclosure.as_ref().is_some_and(|b| !b.contains(&v)) {
                bail!("matched value outside its enclosure");
            }
            values.push((v, enclosure));
        }
        for c in &mut configurations {
            c.branch = SpectralBranch::EvenGround;
            c.external_index = 0;
        }
        if configurations[1..].iter().any(|c| c != &configurations[0]) {
            bail!("matched spectral triple configuration mismatch");
        }
        let interval_order = match (&values[0].1, &values[1].1, &values[2].1) {
            (Some(a), Some(b), Some(c)) => Some(a.upper() < b.lower() && b.upper() < c.lower()),
            _ => None,
        };
        Ok(
            json!({"scope":"matched supplied finite spectral values; same matrix and configuration; index labels and enclosures remain supplied premises; no limiting ordering or simplicity claim",
            "observations":observations,"point_even0_lt_odd0_lt_even1":values[0].0<values[1].0 && values[1].0<values[2].0,
            "enclosed_even0_lt_odd0_lt_even1":interval_order}),
        )
    }

    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct EnergyDistancePremises {
        pub matrix_digest: ContentDigest,
        pub basis_id: String,
        pub metric_id: String,
        pub ground_enclosure: te::ExactBounds,
        pub complementary_eigenvalue_lower: String,
        /// Full-space bound required; an even-sector floor cannot silently be
        /// applied to an unprojected vector with odd energy.
        pub full_space_simple_ground: bool,
        pub provenance: String,
    }
    #[doc(hidden)]
    pub fn energy_distance_bound(
        p: &EnergyDistancePremises,
        rayleigh_upper: &str,
        precision: u32,
    ) -> Result<Value> {
        if !p.matrix_digest.validate()
            || p.basis_id != BASIS
            || p.metric_id != "identity-coefficient-metric-v1"
            || p.provenance.trim().is_empty()
            || !(64..=8192).contains(&precision)
        {
            bail!("energy-distance premise identity");
        }
        let ground = read_bound(&p.ground_enclosure)?;
        let floor = rational(&p.complementary_eigenvalue_lower)?;
        let upper = rational(rayleigh_upper)?;
        if upper < *ground.lower() {
            return Ok(json!({"status":"premise_not_verified",
                "reason":"Rayleigh upper bound is below the declared ground lower bound",
                "premises":p,"rayleigh_upper":rayleigh_upper}));
        }
        if !p.full_space_simple_ground || floor <= *ground.upper() {
            return Ok(
                json!({"status":"unresolved","reason":"full-space simple ground and positive spectral separation required"}),
            );
        }
        let t = (upper - ground.lower()) / Rational::from(&floor - ground.lower());
        let t = t.min(Rational::from(1));
        let cosine = sqrt_bound(&I::point(Rational::from(1) - &t), precision)?;
        let distance = I::point(Rational::from(2)).sub(&cosine.mul(&I::point(Rational::from(2))));
        Ok(
            json!({"status":"conditional_finite_bound","premises":p,"squared_sine_upper":t.to_string(),"phase_aligned_squared_distance_upper":distance.upper().to_string(),
            "scope":"normalized finite vector; caller must establish the full-space simple-ground and complementary-spectrum premises for this exact matrix; no continuum implication"}),
        )
    }
    fn automatic_profile_request(
        s: &RetainedState,
        input: Option<&ExternalResearchInputs>,
    ) -> Result<Option<ProfileRequest>> {
        let Some((i, t, coefficients)) = input.and_then(|i| {
            i.target
                .as_ref()
                .and_then(|t| t.trial_coefficients.as_ref().map(|v| (i, t, v)))
        }) else {
            return Ok(None);
        };
        if coefficients.len() != s.coefficients.len() {
            bail!("profile projection shape mismatch");
        }
        let target = coefficients
            .iter()
            .map(|v| exact(&Float::with_val(i.precision_bits, Float::parse(v)?)))
            .collect::<Result<Vec<_>>>()?;
        let center = target
            .iter()
            .enumerate()
            .fold(Rational::from(0), |sum, (j, v)| {
                sum + Rational::from(
                    v * if j.abs_diff(s.modes).is_multiple_of(2) {
                        1
                    } else {
                        -1
                    },
                )
            });
        let vector = |label: &str, values: Vec<Rational>| ComplexVector {
            label: label.into(),
            imaginary: vec![te::ExactBounds::point("0"); values.len()],
            real: values.into_iter().map(point).collect(),
        };
        let reference = vector(
            "finite_target_reference",
            target
                .iter()
                .map(|x| {
                    if center == 0 {
                        x.clone()
                    } else {
                        Rational::from(x / &center)
                    }
                })
                .collect(),
        );
        let mut request = ProfileRequest {
            basis_id: BASIS.into(),
            lambda_squared: s.cutoff.clone(),
            scope: format!(
                "stored state and finite target projection; functional reference {}",
                if center == 0 {
                    "retains raw target scale because center is zero"
                } else {
                    "has center one"
                }
            ),
            definition_digest: t.definition_digest.clone(),
            profiles: vec![
                vector(
                    "retained_state",
                    s.coefficients.iter().map(exact).collect::<Result<_>>()?,
                ),
                vector("finite_target_projection", target),
            ],
            functional: BoundedFunctional {
                id: format!(
                    "finite-window-target-inner-product:{}",
                    t.definition_digest.0
                ),
                reference,
                denominator: FunctionalDenominator::ReferenceSquaredNorm,
            },
            intervals_per_half: 32,
            precision_bits: s
                .precision
                .max(i.precision_bits)
                .saturating_add(32)
                .min(8192),
        };
        if let Some(trials) = i
            .finite_diagnostics
            .as_ref()
            .and_then(|i| i.complex_trials.as_ref())
        {
            let mut sum = vec![
                (I::point(Rational::from(0)), I::point(Rational::from(0)));
                s.coefficients.len()
            ];
            for (index, part) in std::iter::once(&trials.baseline)
                .chain(&trials.corrections)
                .enumerate()
            {
                let coefficients = complex_coefficients(part, s.coefficients.len())?;
                for (z, v) in sum.iter_mut().zip(&coefficients) {
                    z.0 = checked_interval(z.0.add(&v.0))?;
                    z.1 = checked_interval(z.1.add(&v.1))?;
                }
                request.profiles.push(ComplexVector {
                    label: format!("trial_stage_{index}:{}", part.label),
                    real: sum.iter().map(|v| pack_bound(&v.0)).collect(),
                    imaginary: sum.iter().map(|v| pack_bound(&v.1)).collect(),
                });
            }
        }
        Ok(Some(request))
    }
    fn paired_profile_result(
        id: &str,
        s: &RetainedState,
        input: Option<&ExternalResearchInputs>,
    ) -> Result<Analysis> {
        let request = automatic_profile_request(s, input)?;
        let target = input
            .and_then(|i| i.finite_diagnostics.as_ref())
            .and_then(|i| i.target_profiles.as_ref());
        if request.is_none() && target.is_none() {
            return Ok(empty(
                id,
                s,
                "missing_input",
                "finite target projection or independent profile request required",
            ));
        }
        let paired = request.as_ref().map(analyze_profiles).transpose()?;
        let independent = target.map(analyze_profiles).transpose()?;
        let legacy_eligible = matches!(center_normalized(s,input),Ok(Some((ref x,ref y,_,_))) if x.iter().eq(x.iter().rev()) && y.iter().eq(y.iter().rev()));
        let mut out = if legacy_eligible {
            automatic_profile(id, s, input)?
        } else {
            computed(id, s, json!({}), "certified_finite_enclosure")
        };
        out.result["paired_profiles"] = paired.unwrap_or(Value::Null);
        out.result["target_only_profiles"] = independent.unwrap_or(Value::Null);
        // A pair with an unresolved normalizer remains visible without suppressing
        // the other normalization, or an independent target-only computation.
        if [
            &out.result["paired_profiles"],
            &out.result["target_only_profiles"],
        ]
        .iter()
        .any(|v| {
            v["pairs"]
                .as_array()
                .is_some_and(|rows| rows.iter().any(|r| r["status"] == "unresolved"))
        }) {
            out.rows[0].outcome = "unresolved".into();
        }
        out.scope="directed finite Fourier profile comparisons under center and bounded-functional normalization on both reciprocal halves; no target projection or limiting-operator certificate".into();
        Ok(out)
    }
    fn center_normalized(
        s: &RetainedState,
        input: Option<&ExternalResearchInputs>,
    ) -> Result<Option<CenterNormalized>> {
        let Some((i, target)) = input.and_then(|i| {
            i.target
                .as_ref()
                .and_then(|t| t.trial_coefficients.as_ref())
                .map(|t| (i, t))
        }) else {
            return Ok(None);
        };
        if target.len() != s.coefficients.len() {
            bail!("normalization projection shape mismatch");
        }
        let y = target
            .iter()
            .map(|x| exact(&Float::with_val(i.precision_bits, Float::parse(x)?)))
            .collect::<Result<Vec<_>>>()?;
        let x = s
            .coefficients
            .iter()
            .map(exact)
            .collect::<Result<Vec<_>>>()?;
        let center = |v: &[Rational]| {
            v.iter().enumerate().fold(Rational::from(0), |sum, (j, v)| {
                sum + Rational::from(
                    v * if j.abs_diff(s.modes).is_multiple_of(2) {
                        1
                    } else {
                        -1
                    },
                )
            })
        };
        let a = center(&x);
        let b = center(&y);
        if a == 0 || b == 0 {
            return Err(xc_solver::SolverError::PremiseNotVerified(
                "finite center normalizer is zero; vectors cannot be center normalized".into(),
            )
            .into());
        }
        Ok(Some((
            x.into_iter().map(|x| x / &a).collect(),
            y.into_iter().map(|y| y / &b).collect(),
            a,
            b,
        )))
    }
    fn normalization_result(
        id: &str,
        s: &RetainedState,
        input: Option<&ExternalResearchInputs>,
    ) -> Result<Analysis> {
        let profile_request = automatic_profile_request(s, input)?;
        let functional = profile_request
            .as_ref()
            .map(|r| analyze_profiles_inner(r, false))
            .transpose()?;
        // Functional normalization can be valid when point normalization fails.
        // Retain that result instead of discarding it with the center failure.
        if let Some(value) = functional.as_ref() {
            if value["pairs"].as_array().is_some_and(|rows| {
                rows.iter()
                    .any(|r| r["normalization"] == "center" && r["status"] == "unresolved")
            }) {
                return Ok(computed(
                    id,
                    s,
                    json!({"paired_normalization":value,"reason":"center normalizer includes zero"}),
                    "unresolved",
                ));
            }
        }
        let Some((x, y, a, b)) = center_normalized(s, input)? else {
            return Ok(empty(id,s,"missing_input","requires declared raw error/normalizer bounds or a retained finite target projection"));
        };
        let squared = x.iter().zip(&y).fold(Rational::from(0), |sum, (x, y)| {
            let d = Rational::from(x - y);
            sum + Rational::from(&d * &d)
        });
        let p = s.precision.max(input.unwrap().precision_bits);
        let sqrt = |round| -> Result<String> {
            let mut f = Float::with_val_round(p, &squared, round).0;
            f.sqrt_round(round);
            Ok(exact(&f)?.to_string())
        };
        let norm = te::ExactBounds {
            lower: sqrt(rug::float::Round::Down)?,
            upper: sqrt(rug::float::Round::Up)?,
        };
        let mut out = computed(
            id,
            s,
            json!({"squared_difference":squared.to_string(),
            "normalized_norm_error":norm,"source_center_normalizer":a.to_string(),
            "reference_center_normalizer":b.to_string(),"working_precision_bits":p,
            "paired_normalization":functional}),
            "certified_finite_enclosure",
        );
        out.scope="direct exact squared center-normalized coefficient l2 difference with directed square-root enclosure between stored source and stored finite target projection; excludes continuum reference, source and projection errors".into();
        Ok(out)
    }
    #[cfg(not(feature = "arb"))]
    fn root_result(
        id: &str,
        s: &RetainedState,
        _roots: Option<&RetainedRoots>,
        _certificate: Option<
            &super::super::sector_gap_certificate::PortableCcmSectorGapCertificate,
        >,
    ) -> Result<Analysis> {
        Ok(empty(id,s,"blocked","analytic finite root transfer requires the arb feature and a replayable sector certificate"))
    }
    #[cfg(feature = "arb")]
    fn root_result(
        id: &str,
        s: &RetainedState,
        roots: Option<&RetainedRoots>,
        certificate: Option<&super::super::sector_gap_certificate::PortableCcmSectorGapCertificate>,
    ) -> Result<Analysis> {
        use super::super::sector_gap_certificate::root_budget::{self, Request, Window};
        let (Some(roots), Some(certificate)) = (roots, certificate) else {
            return Ok(empty(id,s,"missing_input","requires retained root outcomes and the run's explicitly requested sector-gap certificate"));
        };
        if s.modes > 512 || roots.dataset.points.len() > 256 || certificate.precision_bits > 65536 {
            return Ok(empty(
                id,
                s,
                "blocked",
                "analytic root transfer exceeds its validated dimension, row or precision limit",
            ));
        }
        let p = s.precision.max(certificate.precision_bits);
        let radius = Rational::from((1, rug::Integer::from(1) << (p / 8).clamp(16, 256)));
        let mut windows = vec![];
        let mut rows = vec![];
        for point in &roots.dataset.points {
            let Some(value) = &point.value else {
                rows.push(Row {
                    label: point.ordinal.to_string(),
                    outcome: "missing_input".into(),
                    result: json!({"source_status":point.source_status}),
                });
                continue;
            };
            let x = exact(&Float::with_val(
                roots.dataset.precision_bits,
                Float::parse(value)?,
            ))?;
            windows.push(Window {
                requested_index: point.ordinal,
                bracket: te::ExactBounds {
                    lower: Rational::from(&x - &radius).to_string(),
                    upper: Rational::from(&x + &radius).to_string(),
                },
                center: x.to_string(),
            });
        }
        if windows.is_empty() {
            return Ok(empty(
                id,
                s,
                "missing_input",
                "no retained root centers available",
            ));
        }
        let request = Request {
            precision_bits: p,
            windows,
        };
        let report = root_budget::analyze(s, certificate, &request, &CancellationToken::new())?;
        for r in &report.rows {
            rows.push(Row {
                label: r.requested_index.to_string(),
                outcome: if r.status == cv::RootStatus::CertifiedLocal {
                    "certified_finite_enclosure"
                } else {
                    "unresolved"
                }
                .into(),
                result: serde_json::to_value(r)?,
            });
        }
        let mut a = computed(
            id,
            s,
            json!({"request":request,"report":report,"window_policy":"center +/- 2^-clamp(precision/8,16,256); no automatic root replacement"}),
            "certified_finite_enclosure",
        );
        a.rows = rows;
        Ok(a)
    }
    #[derive(Clone, Debug, Default, PartialEq, Eq)]
    pub struct ExecutionContext {
        pub policy: super::super::capture_runtime::CaptureResourcePolicy,
        pub certificate_error: Option<String>,
        pub input_preparation_error: Option<String>,
    }
    /// Authenticated primary sources for supplemental analysis. Construction
    /// only reads named immutable payloads; there is no eigensolve fallback.
    #[doc(hidden)]
    pub struct RetainedFiniteSources {
        state: RetainedState,
        matrix: Option<RetainedMatrix<'static>>,
        roots: Option<RetainedRoots>,
    }
    /// A cohort assembled entirely after acquisition. Retained values and
    /// matrix identities are checked, with no source-solving fallback. The
    /// arithmetic gate still makes no error-to-limit or global-index claim.
    #[doc(hidden)]
    pub fn capture_retained_refinement_cohort(
        cohort: &RefinementCohort,
        retained: &[&RetainedFiniteSources],
        cache: &ArtifactCacheContext<'_>,
    ) -> Result<ArtifactExecutionCacheResult<ResearchRecord<Value>>> {
        if retained.len() != cohort.observations.len()
            || cache.write_visibility == xc_cache::CacheVisibility::Public
        {
            bail!("retained cohort source count or private visibility mismatch");
        }
        let mut parents = vec![];
        for (source, observation) in retained.iter().zip(&cohort.observations) {
            let expected = source
                .refinement_observation(observation.configuration.clone(), observation.validity)?;
            if expected.source != observation.source
                || expected.matrix != observation.matrix
                || rational(&expected.value)? != rational(&observation.value)?
            {
                bail!("cohort value or identity differs from its authenticated retained source");
            }
            if observation.enclosure.is_some() {
                bail!("retained point-source cohort cannot authenticate an externally asserted eigenvalue enclosure");
            }
            parents.push(source.state.manifest.clone());
            parents.push(source.matrix.as_ref().unwrap().manifest.clone());
        }
        managed(
            KIND,
            json!({"semantics":SEMANTICS,"diagnostic":"retained_refinement_cohort","input":cohort}),
            &parents,
            cache,
            || {
                let mut result = analyze_refinement_cohort(cohort)?;
                result["source_authentication"]=json!("values and matrix identities replayed from immutable retained primary payloads; branch declarations are not new certificates");
                Ok(result)
            },
            |r| {
                if r["semantics"] != "typed-refinement-cohort-v1" {
                    bail!("retained cohort result semantics mismatch");
                }
                Ok(())
            },
        )
    }
    #[derive(Clone, Debug, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct CohortRoot {
        pub ordinal: usize,
        pub source_status: String,
        pub value: Option<String>,
    }
    #[derive(Clone, Debug, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct CohortObservation {
        pub schema_version: u32,
        pub scope: String,
        pub source_eigenpair: ContentDigest,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub source_matrix: Option<ContentDigest>,
        #[serde(default)]
        pub eigenvalue: String,
        pub source_roots: Option<ContentDigest>,
        pub lambda_squared: String,
        pub n_modes: usize,
        pub precision_bits: u32,
        pub root_precision_bits: Option<u32>,
        pub selection_policy: Option<String>,
        pub acquisition: Value,
        pub global_ordinal_certified: bool,
        pub roots: Vec<CohortRoot>,
    }
    impl RetainedFiniteSources {
        /// The numerical value and matrix identity come from authenticated
        /// retained payloads. Branch labels and acquisition acceptance remain
        /// explicit caller declarations, not new spectral certificates.
        pub fn refinement_observation(
            &self,
            configuration: RefinementConfiguration,
            validity: ObservationValidity,
        ) -> Result<RefinementObservation> {
            let configuration = canonical_configuration(&configuration)?;
            if configuration.observable != RefinementObservable::Eigenvalue
                || configuration.n_modes != self.state.modes
                || configuration.precision_bits != self.state.precision
                || rational(&configuration.lambda_squared)? != rational(&self.state.cutoff)?
                || configuration.basis_id != BASIS
                || configuration.metric_id != "identity-coefficient-metric-v1"
            {
                bail!("refinement configuration differs from retained source");
            }
            let matrix = self.matrix.as_ref().ok_or_else(|| {
                anyhow::anyhow!("refinement observation requires its retained matrix")
            })?;
            Ok(RefinementObservation {
                source: self.state.manifest.content_digest.clone(),
                matrix: matrix.manifest.content_digest.clone(),
                configuration,
                validity,
                value: exact(&Float::with_val(
                    self.state.precision,
                    Float::parse(&self.state.eigenvalue)?,
                ))?
                .to_string(),
                enclosure: None,
            })
        }
        pub fn from_manifests(
            manifests: &[ArtifactManifest],
            include_matrix: bool,
            cache: &ArtifactCacheContext<'_>,
        ) -> Result<Self> {
            let resolver = cache
                .resolver
                .ok_or_else(|| anyhow::anyhow!("retained analysis requires a source resolver"))?;
            let acceptance = cache.acceptance.ok_or_else(|| {
                anyhow::anyhow!("retained analysis requires an acceptance policy")
            })?;
            let find = |kinds: &[&str]| -> Result<Option<&ArtifactManifest>> {
                let found: Vec<_> = manifests
                    .iter()
                    .filter(|m| kinds.contains(&m.key.kind.as_str()))
                    .collect();
                if found.len() > 1 {
                    bail!("ambiguous retained source kind");
                }
                Ok(found.first().copied())
            };
            let read = |m: &ArtifactManifest| -> Result<Vec<u8>> {
                if m.size_bytes > 256 * 1024 * 1024 {
                    bail!("retained supplemental source exceeds the 256 MiB per-payload admission budget");
                }
                Ok(resolver.read_exact_payload(&m.key, &m.content_digest, acceptance)?)
            };
            let em = find(&["ccm_weil_eigenpair"])?
                .ok_or_else(|| anyhow::anyhow!("retained eigenpair manifest missing"))?;
            let state = RetainedState::from_payload(
                em,
                &read(em)?,
                std::slice::from_ref(&em.content_digest),
            )?;
            let matrix = if include_matrix {
                find(&["ccm_tau_matrix"])?
                    .map(|m| -> Result<_> {
                        let matrix = RetainedMatrix::from_payload(
                            m,
                            &read(m)?,
                            std::slice::from_ref(&m.content_digest),
                        )?;
                        matrix.match_state(&state)?;
                        super::super::retained_evidence::matrix_ancestry(
                            &state, &matrix, manifests, cache,
                        )?;
                        Ok(matrix)
                    })
                    .transpose()?
            } else {
                None
            };
            let roots = find(&["ccm_root_refinement", "ccm_root_discovery_window"])?
                .map(|m| -> Result<_> {
                    let secular = find(&["ccm_secular_source"])?.ok_or_else(|| {
                        anyhow::anyhow!("retained secular source manifest missing")
                    })?;
                    RetainedRoots::from_payload(
                        m,
                        &read(m)?,
                        secular,
                        &read(secular)?,
                        &state,
                        &[m.content_digest.clone(), secular.content_digest.clone()],
                    )
                })
                .transpose()?;
            Ok(Self {
                state,
                matrix,
                roots,
            })
        }
        pub fn observation(&self) -> Result<CohortObservation> {
            let roots = self.roots.as_ref();
            Ok(CohortObservation {schema_version:1,
                scope:"authenticated finite stored point roots; identical ordinal labels are not an independent branch-matching or global-index certificate".into(),
                source_eigenpair:self.state.manifest.content_digest.clone(),source_roots:roots.map(|r|r.manifest.content_digest.clone()),
                source_matrix:self.matrix.as_ref().map(|m|m.manifest.content_digest.clone()).or_else(||self.state.manifest.dependencies.iter().find(|d|d.key.kind=="ccm_tau_matrix").map(|d|d.content_digest.clone())),
                eigenvalue:exact(&Float::with_val(self.state.precision,Float::parse(&self.state.eigenvalue)?))?.to_string(),
                lambda_squared:self.state.cutoff.clone(),n_modes:self.state.modes,precision_bits:self.state.precision,
                root_precision_bits:roots.map(|r|r.dataset.precision_bits),selection_policy:self.state.selection_policy.clone(),
                acquisition:roots.map_or(Value::Null,|r|r.acquisition.clone()),global_ordinal_certified:false,
                roots:roots.map_or_else(||Ok(Vec::new()),|r|r.dataset.points.iter().map(|point|->Result<_>{
                    Ok(CohortRoot {ordinal:point.ordinal,source_status:point.source_status.clone(),value:point.value.as_ref().map(|v|->Result<_>{
                        Ok(exact(&Float::with_val(r.dataset.precision_bits,Float::parse(v)?))?.to_string())
                    }).transpose()?})
                }).collect::<Result<Vec<_>>>())?})
        }
        pub fn capture(
            &self,
            id: &str,
            input: Option<&ExternalResearchInputs>,
            research_sources: &[ArtifactManifest],
            cache: &ArtifactCacheContext<'_>,
        ) -> Result<ArtifactExecutionCacheResult<ResearchRecord<Analysis>>> {
            // Bind exactly the sources used by the live producer. Loading a
            // source for another requested group must not change this identity.
            let matrix = matches!(
                id,
                "trial_vector_energy"
                    | "trial_vector_parity"
                    | "directional_error_bound"
                    | "finite_tail_bound"
                    | "spectral_cluster_bound"
            )
            .then_some(self.matrix.as_ref())
            .flatten();
            let roots = matches!(
                id,
                "finite_root_budget" | "directional_error_bound" | "dimension_precision_budget"
            )
            .then_some(self.roots.as_ref())
            .flatten();
            capture(
                id,
                &self.state,
                matrix,
                roots,
                input,
                None,
                research_sources,
                cache,
            )
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn capture(
        id: &str,
        s: &RetainedState,
        m: Option<&RetainedMatrix<'_>>,
        roots: Option<&RetainedRoots>,
        input: Option<&ExternalResearchInputs>,
        certificate: Option<(
            &super::super::sector_gap_certificate::PortableCcmSectorGapCertificate,
            &ArtifactManifest,
        )>,
        parents: &[ArtifactManifest],
        cache: &ArtifactCacheContext<'_>,
    ) -> Result<ArtifactExecutionCacheResult<ResearchRecord<Analysis>>> {
        capture_with_context(
            id,
            s,
            m,
            roots,
            input,
            certificate,
            parents,
            cache,
            &ExecutionContext {
                policy: super::super::capture_runtime::CaptureResourcePolicy::from_environment()?,
                ..Default::default()
            },
        )
    }
    /// One finite diagnostic computed from exactly its arguments, with no
    /// cache, file, environment or clock input. `limited` is the unretained
    /// record to return when a declared resource limit stopped it.
    pub(crate) struct Unit {
        pub(crate) result: Result<Analysis>,
        pub(crate) limited: Option<Analysis>,
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn compute_unit(
        id: &str,
        s: &RetainedState,
        m: Option<&RetainedMatrix<'_>>,
        roots: Option<&RetainedRoots>,
        input: Option<&ExternalResearchInputs>,
        certificate: Option<(
            &super::super::sector_gap_certificate::PortableCcmSectorGapCertificate,
            &ArtifactManifest,
        )>,
        execution: &ExecutionContext,
    ) -> Unit {
        let policy = &execution.policy;
        let limited = std::cell::RefCell::new(None);
        let result = (|| -> Result<Analysis> {
            let _stage =
                super::super::capture_runtime::Stage::new(format!("finite diagnostic {id}"));
            let n = s.coefficients.len() as u128;
            let precision = s
                .precision
                .max(certificate.map_or(0, |(c, _)| c.precision_bits));
            let scalar_bytes = (precision as u128).div_ceil(8) + 256;
            let estimated = (n * 32 + 4096) * scalar_bytes
                + if matches!(
                    id,
                    "directional_error_bound" | "finite_tail_bound" | "spectral_cluster_bound"
                ) {
                    32 * n * n * scalar_bytes
                } else {
                    0
                }
                + if id == "finite_root_budget" && certificate.is_some() {
                    64 * n * n * scalar_bytes
                } else {
                    0
                };
            if estimated > u128::from(policy.maximum_working_bytes) {
                *limited.borrow_mut() = Some(empty(id,s,"blocked","finite diagnostic workspace (including certificate reassembly) exceeds the declared resource policy"));
                bail!("execution resource limit; do not retain");
            }
            if id == "finite_root_budget" && certificate.is_none() {
                if let Some(reason) = &execution.certificate_error {
                    let mut a = empty(
                        id,
                        s,
                        "blocked",
                        &format!("requested sector-gap certificate failed: {reason}"),
                    );
                    a.input_preparation_error = execution.input_preparation_error.clone();
                    return Ok(a);
                }
            }
            let result = compute(id, s, m, roots, input, certificate.map(|(c, _)| c));
            let a = match result {
                Ok(a) => a,
                Err(e) if e.downcast_ref::<ResourceLimited>().is_some() => {
                    *limited.borrow_mut() = Some(empty(id, s, "blocked", &e.to_string()));
                    bail!("execution resource limit; do not retain");
                }
                Err(e)
                    if matches!(
                        e.downcast_ref::<xc_solver::SolverError>(),
                        Some(xc_solver::SolverError::IterationBudgetExhausted(_))
                    ) =>
                {
                    return Ok(empty(id, s, "blocked", &e.to_string()))
                }
                Err(e)
                    if matches!(
                        e.downcast_ref::<xc_solver::SolverError>(),
                        Some(xc_solver::SolverError::PremiseNotVerified(_))
                    ) =>
                {
                    computed(
                        id,
                        s,
                        json!({"reason":e.to_string()}),
                        "premise_not_verified",
                    )
                }
                Err(e) => return Err(e),
            };
            let mut a = a;
            let supplemental = input.and_then(|i| i.finite_diagnostics.as_ref());
            let supplied_energy = if id == "finite_tail_bound" {
                supplemental
                    .and_then(|i| i.projection_energy.as_ref())
                    .map(energy_extensions::analyze_projection)
                    .transpose()?
            } else if id == "trial_vector_energy" {
                supplemental
                    .and_then(|i| i.continuous_energy.as_ref())
                    .map(energy_extensions::analyze_continuous)
                    .transpose()?
            } else {
                None
            };
            if let Some(value) = supplied_energy {
                let label = if id == "finite_tail_bound" {
                    "projection_energy"
                } else {
                    "continuous_energy"
                };
                if a.outcome != "computed" {
                    a.rows.push(Row {
                        label: "primary_analysis".into(),
                        outcome: a.outcome.clone(),
                        result: json!({"reason":a.reason}),
                    });
                    a.result = json!({"primary_status":a.outcome,"primary_reason":a.reason});
                    a.outcome = "computed".into();
                    a.reason = None;
                }
                a.rows.push(Row {
                    label: label.into(),
                    outcome: value["status"].as_str().unwrap_or("unresolved").into(),
                    result: Value::Null,
                });
                a.result[label] = value;
            } else if a.outcome == "computed"
                && matches!(id, "finite_tail_bound" | "trial_vector_energy")
            {
                let label = if id == "finite_tail_bound" {
                    "projection_energy"
                } else {
                    "continuous_energy"
                };
                a.result[label] = json!({"status":"awaiting_source","reason":"no numerical function request supplied; primary acquisition is independent"});
            }
            if a.outcome == "computed"
                && matches!(id, "normalization_error_bound" | "continuous_l1_bound")
            {
                // An explicit analytic witness supplements, rather than hides,
                // the measurements derivable from this run's actual profiles.
                let key = if id == "normalization_error_bound" {
                    "paired_normalization"
                } else {
                    "paired_profiles"
                };
                if a.result.get(key).is_none() {
                    if let Some(request) = automatic_profile_request(s, input)? {
                        a.result[key] =
                            analyze_profiles_inner(&request, id == "continuous_l1_bound")?;
                    }
                }
                if id == "continuous_l1_bound" && a.result.get("target_only_profiles").is_none() {
                    if let Some(request) = input
                        .and_then(|i| i.finite_diagnostics.as_ref())
                        .and_then(|i| i.target_profiles.as_ref())
                    {
                        a.result["target_only_profiles"] = analyze_profiles(request)?;
                    }
                }
            }
            if id == "spectral_cluster_bound" {
                if let Some(extra) = input.and_then(|i| i.finite_diagnostics.as_ref()) {
                    if !extra.matched_spectral_triples.is_empty() {
                        let matrix = m.ok_or_else(|| {
                            anyhow::anyhow!("matched triples require the retained matrix")
                        })?;
                        let triples = extra
                            .matched_spectral_triples
                            .iter()
                            .map(|triple| {
                                if triple
                                    .iter()
                                    .any(|o| o.matrix != matrix.manifest.content_digest)
                                {
                                    bail!("matched triple does not refer to this retained matrix");
                                }
                                analyze_matched_spectral_triple(triple)
                            })
                            .collect::<Result<Vec<_>>>()?;
                        a.rows.push(Row {
                            label: "matched_spectral_triples".into(),
                            outcome: "computed_not_certified".into(),
                            result: json!(triples),
                        });
                    }
                }
            }
            a.input_preparation_error = execution.input_preparation_error.clone();
            if serde_json::to_vec(&a)?.len() as u64 > policy.maximum_output_bytes {
                *limited.borrow_mut() = Some(empty(
                    id,
                    s,
                    "blocked",
                    "finite diagnostic output exceeds the declared output budget",
                ));
                bail!("execution output limit; do not retain");
            }
            Ok(a)
        })();
        Unit {
            result,
            limited: limited.into_inner(),
        }
    }

    /// Exact digest of the retained state a computation reads.
    pub(crate) fn state_content(s: &RetainedState) -> ContentDigest {
        use sha2::{Digest, Sha256};
        let mut hash = Sha256::new();
        hash.update(b"retained-state-entries-v1\0");
        hash.update(
            serde_json::to_vec(&(
                &s.manifest,
                &s.cutoff,
                s.modes,
                s.precision,
                &s.eigenvalue,
                &s.selection_policy,
            ))
            .unwrap_or_default(),
        );
        hash.update((s.coefficients.len() as u64).to_le_bytes());
        for x in &s.coefficients {
            hash.update(x.prec().to_le_bytes());
            hash.update([u8::from(x.is_sign_negative())]);
            match x.to_integer_exp() {
                Some((significand, exponent)) => {
                    let digits = significand.to_digits::<u8>(rug::integer::Order::Lsf);
                    hash.update([1]);
                    hash.update(exponent.to_le_bytes());
                    hash.update((digits.len() as u64).to_le_bytes());
                    hash.update(&digits);
                }
                None => hash.update([if x.is_nan() { 2 } else { 3 }]),
            }
        }
        ContentDigest(format!("{:x}", hash.finalize()))
    }

    /// Claim of a look-ahead result, tried where the computation would run.
    pub(crate) type Prepared<'a> = Box<dyn FnOnce() -> Option<Unit> + 'a>;

    #[allow(clippy::too_many_arguments)]
    pub fn capture_with_context(
        id: &str,
        s: &RetainedState,
        m: Option<&RetainedMatrix<'_>>,
        roots: Option<&RetainedRoots>,
        input: Option<&ExternalResearchInputs>,
        certificate: Option<(
            &super::super::sector_gap_certificate::PortableCcmSectorGapCertificate,
            &ArtifactManifest,
        )>,
        parents: &[ArtifactManifest],
        cache: &ArtifactCacheContext<'_>,
        execution: &ExecutionContext,
    ) -> Result<ArtifactExecutionCacheResult<ResearchRecord<Analysis>>> {
        capture_with_prepared(
            id,
            s,
            m,
            roots,
            input,
            certificate,
            parents,
            cache,
            execution,
            None,
        )
    }

    /// [`capture_with_context`] that tries `prepared` for the computation
    /// step. Every cache access, source binding and validation stays here, in
    /// the same order; only the computation's value may come from `prepared`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn capture_with_prepared(
        id: &str,
        s: &RetainedState,
        m: Option<&RetainedMatrix<'_>>,
        roots: Option<&RetainedRoots>,
        input: Option<&ExternalResearchInputs>,
        certificate: Option<(
            &super::super::sector_gap_certificate::PortableCcmSectorGapCertificate,
            &ArtifactManifest,
        )>,
        parents: &[ArtifactManifest],
        cache: &ArtifactCacheContext<'_>,
        execution: &ExecutionContext,
        prepared: Option<Prepared<'_>>,
    ) -> Result<ArtifactExecutionCacheResult<ResearchRecord<Analysis>>> {
        if !super::super::capture::FINITE_DIAGNOSTICS.contains(&id) {
            bail!("unknown finite capture identifier");
        }
        let mut sources = vec![s.manifest.clone()];
        if let Some(m) = m {
            m.match_state(s)?;
            sources.extend(matrix_ancestry(s, m, parents, cache)?);
            sources.push(m.manifest.clone());
        }
        if let Some(r) = roots {
            if !xc_cache::manifest_depends_on(&r.secular_manifest, &s.manifest)? {
                bail!("root source mismatch");
            }
            sources.extend([r.manifest.clone(), r.secular_manifest.clone()]);
        }
        let admitted = input
            .map(super::super::extended_research::AdmittedInput::new)
            .transpose()?;
        let input_digest = if let (Some(i), Some(admitted)) = (input, &admitted) {
            i.matches(s)?;
            let artifact = super::super::extended_research::capture_external_source_with_parents(
                s, i, parents, cache,
            )?;
            if let Some(m) = artifact.produced_manifest.or(artifact.reused_manifest) {
                sources.push(m);
            }
            Some(admitted.digest().clone())
        } else {
            None
        };
        // Held for the capture: later digests of this certificate reuse it.
        let _admitted_certificate = if let Some((c, m)) = certificate {
            if m.key.kind != "ccm_sector_gap_certificate" {
                bail!("root budget certificate payload/manifest mismatch");
            }
            let Some(admitted) =
                super::super::sector_gap_certificate::AdmittedCertificate::bind(c, m)?
            else {
                bail!("root budget certificate payload/manifest mismatch");
            };
            sources.push(m.clone());
            Some(admitted)
        } else {
            None
        };
        let mut request = json!({"semantics":SEMANTICS,"diagnostic":id,"input_digest":input_digest,
            "certificate_failure":execution.certificate_error.as_ref().map(|_|"requested_certificate_failed"),
            "input_preparation_failure":execution.input_preparation_error.as_ref().map(|_|"input_preparation_failed"),
            "arb_enabled":cfg!(feature="arb"),"streaming_defaults":stream::Options::default(),"convergence_defaults":cv::Options::default(),
            "matrix_policy":"exact rational symmetric part of retained Tau","root_window_policy":"2^-clamp(precision/8,16,256)"});
        request["automatic_recipe"] = json!("retained_primary_then_finite_analysis_v1");
        request["profile_recipe"] = json!(if matches!(
            id,
            "normalization_error_bound" | "continuous_l1_bound"
        ) {
            PROFILE_RECIPE
        } else {
            "paired_center_and_bounded_functional_complex_profiles_v2"
        });
        if matches!(
            id,
            "directional_error_bound" | "finite_tail_bound" | "spectral_cluster_bound"
        ) {
            // A fixed 120 MiB admission once retained a blocked record; admission
            // now follows the declared working-byte budget.
            request["automatic_matrix_admission"] = json!("declared_working_budget_v2");
        }
        if id == "indexed_prolate_comparison" {
            // Older cached records lack this metadata; only this group's identity changes.
            request["indexed_prolate_convention"] = prolate_convention();
        }
        let limited = std::cell::RefCell::new(None);
        let compute = || -> Result<Analysis> {
            let unit = prepared
                .and_then(|claim| claim())
                .unwrap_or_else(|| compute_unit(id, s, m, roots, input, certificate, execution));
            *limited.borrow_mut() = unit.limited;
            unit.result
        };
        let validate = |a: &Analysis| {
            if a.diagnostic != id
                || a.source_precision_bits != s.precision
                || a.scope.is_empty()
                || ![
                    "computed",
                    "missing_input",
                    "not_applicable",
                    "blocked",
                    "awaiting_source",
                    "awaiting_cohort",
                ]
                .contains(&a.outcome.as_str())
                || (a.outcome == "computed" && a.rows.is_empty())
                || (a.outcome != "computed"
                    && (a.reason.as_ref().is_none_or(|r| r.is_empty()) || !a.result.is_null()))
            {
                bail!("invalid finite diagnostic acquisition record");
            }
            Ok(())
        };
        let first = managed(KIND, request.clone(), &sources, cache, compute, validate);
        let limited = limited.into_inner();
        match (first, limited) {
            (Err(_), Some(data)) => {
                // No cached resource failures: a different host may compute it.
                let unretained = ArtifactCacheContext {
                    resolver: None,
                    reference_resolver: None,
                    acceptance: None,
                    ordered_overlays: cache.ordered_overlays.clone(),
                    mode: xc_cache::ArtifactExecutionCacheMode::Disabled,
                    write_on_miss: false,
                    write_visibility: cache.write_visibility,
                    requested_assurance: cache.requested_assurance,
                    certification_failure_policy: cache.certification_failure_policy,
                    production_sink: None,
                };
                managed(KIND, request, &sources, &unretained, || Ok(data), validate)
            }
            (first, _) => first,
        }
    }
}

/// Diagnostics a retained run computes ahead of their turn on one background
/// lane ([`super::capture_runtime::LookAhead`]), and the exact inputs each
/// result is bound to.
///
/// Every listed computation takes neither roots, a certificate nor another
/// diagnostic's output, and reaches only exact rationals, MPFR, Rayon with
/// ordered collection and the solver crate: no Arb/FLINT call (whose
/// per-thread constant caches could make a value depend on the computing
/// thread) and no cache access. Its value is fixed by the inputs recorded in
/// [`Inputs`]: the diagnostic, exact digests of the state, the matrix entries
/// and the external input, and the execution context or options. The only
/// files it may touch are content-bound local research checkpoints.
pub(crate) mod lookahead {
    use super::super::extended_research::{
        extended_compute, input_content_digest, ExtendedAnalysis, ExtensionOptions,
        ExternalResearchInputs,
    };
    use super::super::retained_evidence::RetainedMatrix;
    use super::super::state_geometry::RetainedState;
    use super::finite_capture::{
        compute_unit, matrix_content, state_content, ExecutionContext, Unit,
    };
    use anyhow::Result;
    use std::sync::Arc;
    use xc_cache::{ArtifactManifest, ContentDigest};

    /// Finite diagnostics (`finite_capture`).
    pub(crate) const FINITE: &[&str] = &[
        "continuous_l1_bound",
        "finite_tail_bound",
        "spectral_cluster_bound",
        "trial_vector_energy",
        "trial_vector_parity",
    ];
    /// Retained convergence diagnostics (`extended_research`). Diagnostics
    /// that read or write local checkpoints (such as `operator_cluster`) are
    /// excluded: a look-ahead job reads no cache and writes no file.
    pub(crate) const EXTENDED: &[&str] = &["finite_section_transfer"];

    /// Diagnostics whose computation reads the retained matrix.
    fn reads_matrix(id: &str) -> bool {
        matches!(
            id,
            "trial_vector_energy"
                | "trial_vector_parity"
                | "finite_tail_bound"
                | "spectral_cluster_bound"
                | "finite_section_transfer"
        )
    }

    pub(crate) enum Value {
        Finite(Unit),
        Extended(Result<ExtendedAnalysis>),
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    pub(crate) struct Inputs {
        id: String,
        state: ContentDigest,
        matrix: Option<(ArtifactManifest, ContentDigest)>,
        input: Option<ContentDigest>,
        execution: Option<ExecutionContext>,
        options: Option<ExtensionOptions>,
    }

    impl Inputs {
        fn of(
            id: &str,
            s: &RetainedState,
            m: Option<&RetainedMatrix<'_>>,
            input: Option<&ExternalResearchInputs>,
        ) -> Result<Self> {
            Ok(Self {
                id: id.to_owned(),
                state: state_content(s),
                matrix: m.map(|m| (m.manifest.clone(), matrix_content(m, false))),
                input: input.map(input_content_digest).transpose()?,
                execution: None,
                options: None,
            })
        }
    }

    pub(crate) type Lane = super::super::capture_runtime::LookAhead<Inputs, Value>;
    type Job = Box<dyn FnOnce() -> Value + Send>;

    /// Jobs for `ids`, in order, from owned copies of exactly the arguments
    /// the serial capture would pass: `finite_input` and `execution` for
    /// finite diagnostics, `extended_input` and the effective `options` for
    /// the others.
    pub(crate) fn jobs(
        ids: &[&str],
        s: &RetainedState,
        m: &RetainedMatrix<'_>,
        finite_input: Option<&ExternalResearchInputs>,
        execution: &ExecutionContext,
        extended_input: Option<&ExternalResearchInputs>,
        options: &ExtensionOptions,
    ) -> Result<Vec<(String, Inputs, Job)>> {
        let state = Arc::new(s.clone());
        let matrix = Arc::new(m.to_owned_entries());
        // One copy when both arguments are the same input.
        let shared =
            matches!((finite_input, extended_input), (Some(a), Some(b)) if std::ptr::eq(a, b));
        let finite_input = finite_input.map(|i| Arc::new(i.clone()));
        let extended_input = if shared {
            finite_input.clone()
        } else {
            extended_input.map(|i| Arc::new(i.clone()))
        };
        let execution = Arc::new(execution.clone());
        let options = Arc::new(options.clone());
        // Digest each input once; the keys share them.
        let finite = Inputs::of("", &state, Some(&matrix), finite_input.as_deref())?;
        let extended = Inputs {
            input: extended_input
                .as_deref()
                .map(input_content_digest)
                .transpose()?,
            ..finite.clone()
        };
        let mut jobs = Vec::new();
        for &id in ids {
            let matrix = reads_matrix(id).then(|| matrix.clone());
            let state = state.clone();
            let owned = id.to_owned();
            let (key, job): (Inputs, Job) = if FINITE.contains(&id) {
                let (input, execution) = (finite_input.clone(), execution.clone());
                let key = Inputs {
                    id: owned.clone(),
                    matrix: finite.matrix.clone().filter(|_| reads_matrix(id)),
                    execution: Some((*execution).clone()),
                    ..finite.clone()
                };
                let job = Box::new(move || {
                    Value::Finite(compute_unit(
                        &owned,
                        &state,
                        matrix.as_deref(),
                        None,
                        input.as_deref(),
                        None,
                        &execution,
                    ))
                });
                (key, job)
            } else if EXTENDED.contains(&id) {
                let (input, options) = (extended_input.clone(), options.clone());
                let key = Inputs {
                    id: owned.clone(),
                    matrix: extended.matrix.clone().filter(|_| reads_matrix(id)),
                    options: Some((*options).clone()),
                    ..extended.clone()
                };
                let job = Box::new(move || {
                    Value::Extended(extended_compute(
                        &owned,
                        &state,
                        matrix.as_deref(),
                        None,
                        &options,
                        input.as_deref(),
                        input.as_deref(),
                        None,
                        false,
                    ))
                });
                (key, job)
            } else {
                continue;
            };
            jobs.push((id.to_owned(), key, job));
        }
        Ok(jobs)
    }

    /// The look-ahead result of a finite diagnostic called with exactly these
    /// arguments, if the lane holds one; `None` means compute inline.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn claim_finite(
        lane: &Lane,
        id: &str,
        s: &RetainedState,
        m: Option<&RetainedMatrix<'_>>,
        roots: bool,
        input: Option<&ExternalResearchInputs>,
        certificate: bool,
        execution: &ExecutionContext,
    ) -> Option<Unit> {
        let value = lane.claim(id, |key| {
            FINITE.contains(&id)
                && !roots
                && !certificate
                && Inputs::of(id, s, m, input).is_ok_and(|actual| {
                    key == &Inputs {
                        execution: Some(execution.clone()),
                        ..actual
                    }
                })
        })?;
        match value {
            Value::Finite(unit) => Some(unit),
            Value::Extended(_) => None,
        }
    }

    /// The look-ahead result of a retained convergence diagnostic computed
    /// from exactly these arguments with no roots, certificate, directional
    /// source or exceeded row budget (`plain`); `None` means compute inline.
    pub(crate) fn claim_extended(
        lane: &Lane,
        id: &str,
        s: &RetainedState,
        m: Option<&RetainedMatrix<'_>>,
        input: Option<&ExternalResearchInputs>,
        options: &ExtensionOptions,
        plain: bool,
    ) -> Option<Result<ExtendedAnalysis>> {
        let value = lane.claim(id, |key| {
            EXTENDED.contains(&id)
                && plain
                && Inputs::of(id, s, m, input).is_ok_and(|actual| {
                    key == &Inputs {
                        options: Some(options.clone()),
                        ..actual
                    }
                })
        })?;
        match value {
            Value::Extended(result) => Some(result),
            Value::Finite(_) => None,
        }
    }
}
