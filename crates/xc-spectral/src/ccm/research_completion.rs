//! Reproducible preparation, comparisons and finite signed-measure models.
//! Every external convention and borrowed input stays in the source artifact.
use super::{
    convergence_capture::OperatorAction, extended_research::*, retained_evidence::*,
    state_geometry::RetainedState,
};
use anyhow::{bail, Result};
use rug::Float;
use serde::{Deserialize, Serialize};
use xc_cache::ContentDigest;
use xc_numerics::prefix::lossless_decimal as dec;

#[path = "research_completion/comparison_math.rs"]
mod comparison_math;
#[path = "research_completion/consistency_math.rs"]
mod consistency_math;
#[path = "research_completion/preparation_points.rs"]
mod preparation_points;
#[path = "research_completion/tail_form_math.rs"]
mod tail_form_math;
pub const DIAGNOSTICS: &[&str] = &[
    "capture_preflight",
    "consistency",
    "configuration_comparison",
    "band_reconstruction",
    "transform_enclosure",
];
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CompletionInputs {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub atom_analysis: Option<super::atom_research::AtomAnalysisPolicy>,
    #[serde(default)]
    pub independent_actions: Vec<OperatorAction>,
    #[serde(default)]
    pub response_checks: Vec<ResponseCheck>,
    #[serde(default)]
    pub comparisons: Vec<ComparisonSnapshot>,
    pub band: Option<SignedBandModel>,
    pub contour: Option<ContourPolicy>,
    pub sector_certificate: Option<super::sector_gap_certificate::PortableCcmSectorGapCertificate>,
    #[serde(default)]
    pub preparation_notes: Vec<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ResponseCheck {
    pub ordinal: usize,
    pub t: String,
    pub source_digest: ContentDigest,
    pub branch: String,
    pub coordinate: String,
    pub derivative_parameter: String,
    pub activation_convention: String,
    pub fixed_velocity: Option<String>,
    pub support_velocity: Option<String>,
    pub total_velocity: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ComparisonSnapshot {
    pub state: super::convergence_capture::ComparisonState,
    pub selection_policy: String,
    pub assembly_policy: String,
    pub quadrature_policy: String,
    pub root_branch: String,
    pub root_coordinate: String,
    #[serde(default)]
    pub roots: Vec<EvaluationPoint>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BandAtom {
    pub coordinate: String,
    pub signed_weight: String,
    pub family: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SignedBandModel {
    pub degree: usize,
    pub coordinate: String,
    pub definition_digest: ContentDigest,
    pub atoms: Vec<BandAtom>,
    pub coverage: String,
    pub hypotheses: Vec<String>,
    pub borrowed_inputs: Vec<String>,
    pub input_energy: Option<String>,
    #[serde(default)]
    pub scoring_roots: Vec<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ContourPolicy {
    pub left: String,
    pub right: String,
    pub bottom: String,
    pub top: String,
    pub maximum_depth: u32,
    pub maximum_segments: usize,
}
impl CompletionInputs {
    pub fn validate(&self, dim: usize, p: u32) -> Result<()> {
        super::retained_evidence::precision(p)?;
        if dim == 0 || dim > 16385 || dim.is_multiple_of(2) {
            bail!("invalid retained source dimension");
        }
        if self.independent_actions.len() > 64
            || self.response_checks.len() > 100000
            || self.comparisons.len() > 64
            || self.preparation_notes.len() > 128
        {
            bail!("completion input budget exceeded");
        }
        for a in &self.independent_actions {
            if a.action.len() != dim
                || !a.source_digest.validate()
                || a.convention.is_empty()
                || a.label.is_empty()
            {
                bail!("invalid independent action");
            }
            for x in &a.action {
                scalar(x, p)?;
            }
        }
        for r in &self.response_checks {
            if r.ordinal == 0
                || !r.source_digest.validate()
                || r.branch.is_empty()
                || r.activation_convention.is_empty()
            {
                bail!("invalid response check binding");
            }
            for x in std::iter::once(&r.t)
                .chain(r.fixed_velocity.iter())
                .chain(r.support_velocity.iter())
                .chain(r.total_velocity.iter())
            {
                scalar(x, p)?;
            }
        }
        for c in &self.comparisons {
            if c.selection_policy.is_empty()
                || c.assembly_policy.is_empty()
                || c.quadrature_policy.is_empty()
                || c.roots.len() > 100000
            {
                bail!("comparison policies absent");
            }
            // Comparisons may vary C, N or P. Validate the comparison at its own precision/dimension.
            let a = &c.state;
            precision(a.precision_bits)?;
            if a.n_modes > 8192
                || a.coefficients.len() != 2 * a.n_modes + 1
                || !a.source_digest.validate()
                || !a.matrix_digest.validate()
                || scalar(&a.lambda_squared, a.precision_bits)? <= 1
            {
                bail!("invalid comparison source shape");
            }
            for v in a.coefficients.iter().chain(std::iter::once(&a.eigenvalue)) {
                scalar(v, a.precision_bits)?;
            }
            if !a.matrix.is_empty() {
                let temp = super::convergence_capture::RunOnceInputs {
                    comparison: Some(a.clone()),
                    ..Default::default()
                };
                temp.validate(2 * a.n_modes + 1, p.max(a.precision_bits))?;
            }
            let mut ordinals = std::collections::BTreeSet::new();
            for r in &c.roots {
                if r.ordinal == 0 || !ordinals.insert(r.ordinal) {
                    bail!("duplicate comparison ordinal");
                }
                if let Some(v) = &r.value {
                    scalar(v, a.precision_bits)?;
                }
            }
        }
        if let Some(a) = &self.atom_analysis {
            a.validate(p)?;
        }
        if let Some(b) = &self.band {
            if b.degree == 0
                || b.degree > 2048
                || b.atoms.len()
                    > self
                        .atom_analysis
                        .as_ref()
                        .map_or(200000, |a| a.maximum_atoms)
                || b.atoms.len() < b.degree
                || !b.definition_digest.validate()
                || b.coverage.is_empty()
                || b.coordinate.is_empty()
                || b.hypotheses.is_empty()
            {
                bail!("invalid signed band model or coverage");
            }
            for a in &b.atoms {
                scalar(&a.coordinate, p)?;
                scalar(&a.signed_weight, p)?;
                if a.family.is_empty() {
                    bail!("band atom family absent");
                }
            }
            if !b.scoring_roots.is_empty() {
                if b.scoring_roots.len() != b.degree {
                    bail!("scoring roots must be empty or match the band degree exactly");
                }
                let roots = coeffs(&b.scoring_roots, p)?;
                if roots.windows(2).any(|v| v[0] >= v[1]) {
                    bail!(
                        "scoring roots must be strictly ascending in the declared band coordinate"
                    );
                }
            }
            for x in b.input_energy.iter().chain(b.scoring_roots.iter()) {
                scalar(x, p)?;
            }
        }
        if let Some(c) = &self.contour {
            if scalar(&c.left, p)? >= scalar(&c.right, p)?
                || scalar(&c.bottom, p)? >= scalar(&c.top, p)?
                || c.maximum_depth > 40
                || c.maximum_segments < 4
                || c.maximum_segments > 100000
            {
                bail!("invalid contour policy");
            }
        }
        if let Some(c) = &self.sector_certificate {
            if dim == 0 || c.n_modes != (dim - 1) / 2 {
                bail!("certificate source dimension mismatch");
            }
        }
        Ok(())
    }
}

/// A reusable, non-executable reference. It is independent of an eigenpair;
/// preparation binds derived values to the exact retained state afterwards.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferencePreparation {
    pub schema_version: u32,
    pub finite_reference: Option<ResearchInputs>,
    pub sampled_reference: Option<SampledReference>,
    pub lambda_squared: String,
    pub precision_bits: u32,
    pub definition_digest: ContentDigest,
    pub approximation_scope: String,
    #[serde(default)]
    pub weighted_atoms: Vec<WeightedAtom>,
    pub atom_coordinate: Option<String>,
    pub atom_coverage: Option<String>,
    pub tail_form: Option<super::convergence_capture::TailForm>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tail_recipe: Option<TailFormRecipe>,
    pub completion: Option<CompletionInputs>,
}
/// Ascending monomial coefficients in the declared atom coordinate.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TailFormRecipe {
    pub basis_polynomials: Vec<Vec<String>>,
    pub tail_correction: Option<Vec<String>>,
    pub hypotheses: Vec<String>,
}
impl ReferencePreparation {
    pub fn from_file(path: &std::path::Path) -> Result<Self> {
        if std::fs::metadata(path)?.len() > 64 * 1024 * 1024 {
            bail!("reference preparation file exceeds 64 MiB");
        }
        Self::from_bytes(path, &std::fs::read(path)?)
    }
    /// Decode a frozen byte snapshot; path resolves separately hashed atom chunks.
    pub fn from_bytes(path: &std::path::Path, bytes: &[u8]) -> Result<Self> {
        if bytes.len() > 64 * 1024 * 1024 {
            bail!("reference preparation exceeds 64 MiB");
        }
        let mut value: Self = serde_json::from_slice(bytes)?;
        super::atom_research::expand_tables(
            path,
            &mut value.weighted_atoms,
            value.completion.as_mut(),
            value.precision_bits,
        )?;
        Ok(value)
    }
    pub fn prepare(
        &self,
        state: &RetainedState,
        roots: Option<&RetainedRoots>,
    ) -> Result<ExternalResearchInputs> {
        let _stage = super::capture_runtime::Stage::new("external reference preparation");
        if self.schema_version != 1
            || !self.definition_digest.validate()
            || self.approximation_scope.is_empty()
            || self.lambda_squared != state.cutoff
        {
            bail!("reference preparation identity mismatch");
        }
        precision(self.precision_bits)?;
        let mut source_precision = self.precision_bits.max(state.precision);
        if let Some(f) = &self.finite_reference {
            f.validate()?;
            source_precision = source_precision.max(f.reference.precision_bits);
            for basis in &f.basis {
                source_precision = source_precision.max(basis.precision_bits);
            }
        }
        if let Some(roots) = roots {
            if !xc_cache::manifest_depends_on(&roots.secular_manifest, &state.manifest)? {
                bail!("reference preparation roots belong to another state");
            }
            source_precision = source_precision.max(roots.dataset.precision_bits);
        }
        let p = source_precision.saturating_add(64);
        precision(p)?;
        let promotion = preparation_points::Promotion {
            source: self.precision_bits,
            target: p,
        };
        let mut input: ExternalResearchInputs = serde_json::from_value(
            serde_json::json!({"schema_version":1,"source_eigenpair":state.manifest.content_digest,"lambda_squared":state.cutoff,"n_modes":state.modes,"precision_bits":p,"convention_id":"external_reference_preparation_v2","definition_digest":self.definition_digest,"approximation_scope":self.approximation_scope}),
        )?;
        input.target = self.sampled_reference.clone();
        input.atoms = self.weighted_atoms.clone();
        input.atom_coordinate = self.atom_coordinate.clone();
        input.atom_coverage = self.atom_coverage.clone();
        if let Some(sample) = &mut input.target {
            promotion.sampled(sample)?;
        }
        promotion.atoms(&mut input.atoms)?;
        input.run_once = Some(super::convergence_capture::RunOnceInputs {
            tail_form: self.tail_form.clone(),
            completion: self.completion.clone(),
            ..Default::default()
        });
        if let Some(form) = &mut input.run_once.as_mut().unwrap().tail_form {
            promotion.form(form)?;
        }
        if let Some(completion) = &mut input.run_once.as_mut().unwrap().completion {
            promotion.completion(completion)?;
        }
        if let Some(recipe) = &self.tail_recipe {
            let mut recipe = recipe.clone();
            promotion.recipe(&mut recipe)?;
            if self.tail_form.is_some() {
                bail!("choose either an explicit tail form or a tail-form recipe");
            }
            let coverage = self
                .atom_coverage
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("tail recipe atom coverage absent"))?;
            input.run_once.as_mut().unwrap().tail_form = Some(prepare_tail_form(
                self.definition_digest.clone(),
                &recipe.basis_polynomials,
                &input.atoms,
                recipe.tail_correction.as_deref(),
                coverage,
                &recipe.hypotheses,
                p,
            )?);
        }
        if let Some(f) = &self.finite_reference {
            f.validate()?;
            if f.reference.lambda_squared != state.cutoff {
                bail!("finite reference cutoff mismatch");
            }
            let coefficients = f.reference.values(p)?;
            if coefficients
                .iter()
                .zip(coefficients.iter().rev())
                .any(|(a, b)| a != b)
            {
                bail!("automatic real-even reference preparation requires even coefficients");
            }
            let center = super::retained_evidence::center(&coefficients, p)?;
            if center == 0 {
                bail!("finite reference center is zero");
            }
            let normalized = coefficients
                .iter()
                .map(|v| Float::with_val(p, v) / &center)
                .collect::<Vec<_>>();
            let intervals = (state.modes.max(coefficients.len() / 2) * 16).clamp(256, 131072);
            if input.target.is_none() {
                let mut values = Vec::with_capacity(intervals + 1);
                let mut basis_values = vec![Vec::with_capacity(intervals + 1); f.basis.len()];
                let basis = f
                    .basis
                    .iter()
                    .map(|b| {
                        if b.lambda_squared != state.cutoff {
                            bail!("basis cutoff mismatch");
                        }
                        let values = b.values(p)?;
                        if values.iter().ne(values.iter().rev()) {
                            bail!("automatic real-even reference preparation requires even basis coefficients");
                        }
                        Ok(values)
                    })
                    .collect::<Result<Vec<_>>>()?;
                for j in 0..=intervals {
                    // The node is x/L=j/(2m); evaluate in that normalized
                    // coordinate directly so cutoff rounding cannot move it.
                    let x = Float::with_val(p, j) / (2 * intervals);
                    let unit_length = Float::with_val(p, 1);
                    // Normalization defines the center exactly. Summing huge
                    // normalized terms can erase its unit value by cancellation.
                    values.push(if j == 0 {
                        "1".into()
                    } else {
                        dec(&super::extended_research::evaluate(&normalized, &x, &unit_length, p).0)
                    });
                    for (v, out) in basis.iter().zip(&mut basis_values) {
                        out.push(dec(&if j == 0 {
                            super::retained_evidence::center(v, p)?
                        } else {
                            super::extended_research::evaluate(v, &x, &unit_length, p).0
                        }));
                    }
                }
                input.target = Some(SampledReference {
                    definition_digest: self.definition_digest.clone(),
                    evaluation_policy:
                        "analytic supplied finite Fourier reference on uniform log-coordinate nodes"
                            .into(),
                    approximation_scope: f.reference.approximation_scope.clone(),
                    intervals,
                    values,
                    basis_values,
                    fixed_second_component: None,
                    raw_normalizer: dec(&center),
                    trial_coefficients: None,
                });
            }
            if let Some(roots) = roots {
                let l = finite_math::rounded_log_cutoff(&state.cutoff, p)?;
                // This is the transform of the explicitly finite reference, whose
                // exterior is zero by definition, not an omitted infinite target tail.
                for point in &roots.dataset.points {
                    let Some(t) = &point.value else { continue };
                    let t_value = Float::with_val(p, scalar(t, roots.dataset.precision_bits)?);
                    let mut reference_state = state.clone();
                    reference_state.coefficients = normalized.clone();
                    reference_state.modes = normalized.len() / 2;
                    let (value, derivative, _, _) = transform_terms(&reference_state, &t_value, p)?;
                    let norm = normalized
                        .iter()
                        .fold(Float::with_val(p, 0), |n, x| n.hypot(x));
                    let scale = norm * l.clone().sqrt();
                    if !scale.is_finite() || scale <= 0 {
                        bail!("finite reference transform scale exceeds supported exponent range");
                    }
                    let jet = Jet {
                        value: dec(&(value * &scale)),
                        derivative: dec(&(derivative * &scale)),
                    };
                    input.reference_jets.push(ReferenceJet {
                        matched_root_ordinal: None,
                        ordinal: point.ordinal,
                        t: dec(&t_value),
                        reference_window: jet.clone(),
                        reference_full: jet,
                        exterior_tail: Jet {
                            value: "0".into(),
                            derivative: "0".into(),
                        },
                        endpoint_tail_part: None,
                        fitted_interior_parts: vec![],
                        error_normalization: None,
                        source_value_error: None,
                        tail_value_error: None,
                        source_derivative_error: None,
                        root_separation_radius: None,
                    });
                }
            }
            input.run_once.as_mut().unwrap().producer_notes.push("automatically prepared finite-reference samples and transform jets; exterior zero belongs only to the declared compact finite reference".into());
        }
        input.validate()?;
        Ok(input)
    }
}

pub(crate) fn completion(i: Option<&ExternalResearchInputs>) -> Option<&CompletionInputs> {
    i.and_then(|i| i.run_once.as_ref())
        .and_then(|i| i.completion.as_ref())
}
fn preflight(
    s: &RetainedState,
    m: Option<&RetainedMatrix<'_>>,
    roots: Option<&RetainedRoots>,
    o: &ExtensionOptions,
    i: Option<&ExternalResearchInputs>,
) -> Result<ExtendedAnalysis> {
    let mut r = report("capture_preflight", s, o);
    let c = completion(i);
    let policy = super::capture_runtime::CaptureResourcePolicy::from_environment()?;
    for (name, value) in [
        (
            "maximum_working_bytes",
            o.maximum_working_bytes
                .unwrap_or(policy.maximum_working_bytes),
        ),
        ("maximum_output_bytes", o.maximum_estimated_output_bytes),
        (
            "root_count",
            roots.map_or(0, |r| r.dataset.points.len()) as u64,
        ),
        (
            "estimated_complement_workspace_bytes",
            (s.coefficients.len() as u64)
                .saturating_pow(2)
                .saturating_mul(3 * (u64::from(o.working_precision_bits).div_ceil(8) + 64)),
        ),
    ] {
        put(
            &mut r.values,
            name,
            &Float::with_val(o.working_precision_bits, value),
        );
    }
    let requirements = [
        ("primary_state", true),
        ("retained_matrix", m.is_some()),
        ("retained_roots", roots.is_some()),
        ("sampled_reference", i.is_some_and(|i| i.target.is_some())),
        (
            "reference_transform_jets",
            i.is_some_and(|i| !i.reference_jets.is_empty()),
        ),
        ("weighted_atoms", i.is_some_and(|i| !i.atoms.is_empty())),
        (
            "independent_prime_action",
            c.is_some_and(|c| !c.independent_actions.is_empty()),
        ),
        (
            "comparison_cohort",
            c.is_some_and(|c| !c.comparisons.is_empty()),
        ),
        (
            "signed_band_model",
            c.is_some_and(|c| c.band.is_some()) || polynomial_tail(i).is_some(),
        ),
        (
            "source_assembly_certificate",
            c.is_some_and(|c| c.sector_certificate.is_some()),
        ),
        (
            "tail_bilinear_model",
            i.and_then(|i| i.run_once.as_ref())
                .is_some_and(|i| i.tail_form.is_some()),
        ),
    ];
    for (idx, (name, available)) in requirements.iter().enumerate() {
        let mut rr = row(idx + 1, *name);
        put(
            &mut rr.values,
            "available",
            &Float::with_val(o.working_precision_bits, u32::from(*available)),
        );
        if !available {
            rr.outcome = "missing_input".into();
            rr.notes.push("supply an identified retained source or external numeric reference; do not repeat the primary solve".into());
        }
        r.rows.push(rr);
    }
    if r.rows.iter().any(|r| r.outcome != "point_measurement") {
        r.outcome = "partial_unresolved".into();
        r.reason = Some(
            "prerequisite coverage; unavailable optional sources do not invalidate primary results"
                .into(),
        );
    }
    r.convention="preflight source inventory and conservative workspace estimates; not a numerical acceptance verdict".into();
    Ok(r)
}
fn comparisons(
    s: &RetainedState,
    m: Option<&RetainedMatrix<'_>>,
    roots: Option<&RetainedRoots>,
    o: &ExtensionOptions,
    i: Option<&ExternalResearchInputs>,
) -> Result<ExtendedAnalysis> {
    use super::retained_evidence::{
        finite_math::{abs, narrow},
        transform_math,
    };
    use xc_numerics::mpfr_interval::MpfrInterval as I;
    let mut r = report("configuration_comparison", s, o);
    let Some(completion) = completion(i).filter(|c| !c.comparisons.is_empty()) else {
        return Ok(missing(
            r,
            "comparison snapshots or a verified retained-source cohort required",
        ));
    };
    let p = o.working_precision_bits;
    let mut multiplicities = std::collections::BTreeMap::new();
    let canonical = |s: &str| -> Result<String> {
        Ok(xc_core::DecimalLiteral::new(s)?
            .canonical()?
            .as_str()
            .to_owned())
    };
    let coordinate = |c: &ComparisonSnapshot| -> Result<_> {
        Ok((
            canonical(&c.state.lambda_squared)?,
            c.state.n_modes,
            c.state.precision_bits,
            c.assembly_policy.clone(),
            c.quadrature_policy.clone(),
            c.selection_policy.clone(),
        ))
    };
    for c in &completion.comparisons {
        *multiplicities.entry(coordinate(c)?).or_insert(0usize) += 1;
    }
    let cutoff = canonical(&s.cutoff)?;
    let branch = roots
        .map(|r| serde_json::to_string(&r.acquisition))
        .transpose()?
        .unwrap_or_default();
    for (idx, c) in completion.comparisons.iter().enumerate() {
        let mut rr = row(idx + 1, "independent_configuration");
        let a = &c.state;
        rr.notes.push(format!(
            "source {}; matrix {}; selection {}; assembly {}; quadrature {}; root branch {}",
            a.source_digest.0,
            a.matrix_digest.0,
            c.selection_policy,
            c.assembly_policy,
            c.quadrature_policy,
            c.root_branch
        ));
        let other_cutoff = canonical(&a.lambda_squared)?;
        if multiplicities[&coordinate(c)?] > 1 {
            rr.outcome = "unresolved_denominator".into();
            rr.notes
                .push("ambiguous duplicate policy configuration; no source selected".into());
            r.rows.push(rr);
            continue;
        }
        if p < a.precision_bits {
            rr.outcome = "budget_limited".into();
            rr.notes
                .push("working precision below comparison source".into());
            r.rows.push(rr);
            continue;
        }
        let same_basis = other_cutoff == cutoff;
        let matched_branch =
            c.root_coordinate == "mellin_t" && c.root_branch == branch && roots.is_some();
        let joins = if matched_branch {
            roots
                .unwrap()
                .dataset
                .points
                .iter()
                .filter(|point| {
                    point.value.is_some()
                        && c.roots
                            .iter()
                            .any(|v| v.ordinal == point.ordinal && v.value.is_some())
                })
                .count()
        } else {
            0
        };
        let output = (8192u64 + (32 + 12 * joins as u64) * (u64::from(p) / 3 + 128))
            * completion.comparisons.len() as u64;
        let scratch = (8 * (s.coefficients.len() + a.coefficients.len()) as u64 + 256)
            * (u64::from(p + 4096).div_ceil(8) + 64);
        if output > o.maximum_estimated_output_bytes
            || o.maximum_working_bytes.is_some_and(|limit| scratch > limit)
        {
            rr.outcome = "budget_limited".into();
            rr.notes.push(
                "comparison interval scratch or joined-root output exceeds explicit budget".into(),
            );
            r.rows.push(rr);
            continue;
        }
        let mut other_state = s.clone();
        other_state.cutoff = a.lambda_squared.clone();
        other_state.modes = a.n_modes;
        other_state.precision = a.precision_bits;
        other_state.coefficients = coeffs(&a.coefficients, a.precision_bits)?;
        other_state.eigenvalue = a.eigenvalue.clone();
        other_state.selection_policy = Some(c.selection_policy.clone());
        let mut measured = None;
        for guard in [64, 128, 256, 512, 1024, 2048, 4096] {
            let Some(mut base) = comparison_math::measure_at(s, a, m, same_basis, p, guard)? else {
                continue;
            };
            if base.zero_comparison {
                measured = Some(base);
                break;
            }
            let work = p + guard;
            let mut accepted = true;
            if matched_branch {
                for point in &roots.unwrap().dataset.points {
                    if let (Some(t), Some(other)) = (
                        &point.value,
                        c.roots
                            .iter()
                            .find(|v| v.ordinal == point.ordinal)
                            .and_then(|v| v.value.as_ref()),
                    ) {
                        let t = scalar(t, roots.unwrap().dataset.precision_bits)?;
                        let other = scalar(other, a.precision_bits)?;
                        base.values.insert(
                            format!("root_{}_signed_difference", point.ordinal),
                            I::from_float(&t, work)?.sub(&I::from_float(&other, work)?),
                        );
                        let Some(current) = transform_math::measure_at_root(s, &t, p, guard)?
                        else {
                            accepted = false;
                            break;
                        };
                        let Some(comparison) =
                            transform_math::measure_at_root(&other_state, &t, p, guard)?
                        else {
                            accepted = false;
                            break;
                        };
                        for (label, x, y) in [
                            ("value", current.value, comparison.value),
                            ("slope", current.derivative, comparison.derivative),
                        ] {
                            base.values.insert(
                                format!("root_{}_transform_{label}_difference", point.ordinal),
                                x.sub(&y),
                            );
                        }
                    }
                }
            }
            if !accepted {
                continue;
            }
            for value in base.values.values() {
                let magnitude = if value.contains_zero() {
                    Float::with_val(work, 1)
                } else {
                    abs(value)?.upper().clone()
                };
                if !narrow(std::slice::from_ref(value), &magnitude, p)? {
                    accepted = false;
                    break;
                }
            }
            if accepted {
                measured = Some(base);
                break;
            }
        }
        if let Some(measured) = measured {
            for (name, value) in measured.values {
                save_arithmetic_enclosure(&mut rr.values, &name, &value, p)?;
            }
            put(
                &mut rr.values,
                "arithmetic_precision_bits",
                &Float::with_val(p, measured.precision),
            );
            if measured.zero_comparison {
                rr.outcome = "unresolved_denominator".into();
                rr.notes.push("comparison coefficient vector is exactly zero at its declared precision; normalization and transform comparisons withheld".into());
            }
        } else {
            rr.outcome = "cancellation_limited".into();
            rr.notes.push(
                "comparison arithmetic unresolved within 4096 guard bits; measurements withheld"
                    .into(),
            );
        }
        if !same_basis || a.n_modes > s.modes {
            rr.notes.push("different exact cutoffs or larger comparison support: same-basis overlap and matrix residual withheld".into());
        }
        if !matched_branch {
            rr.notes.push(
                "root branch/coordinate mismatch or absence: ordinal differences withheld".into(),
            );
        }
        r.rows.push(rr);
    }
    if r.rows.iter().any(|row| row.outcome != "point_measurement") {
        r.outcome = "partial_unresolved".into();
        r.reason = Some("one or more comparison configurations were withheld".into());
    }
    r.convention="declared C/N/P/quadrature cohorts at original stored precision; scaled overlap and entrywise shifted residual enclosures; no fitted rate or substituted parent prefix; root joins require equal acquisition branch and ordinal; external source/selection hypotheses unverified".into();
    Ok(r)
}

pub(crate) fn band_single(
    s: &RetainedState,
    o: &ExtensionOptions,
    i: Option<&ExternalResearchInputs>,
) -> Result<ExtendedAnalysis> {
    let mut r = report("band_reconstruction", s, o);
    let Some(b) = completion(i).and_then(|c| c.band.as_ref()) else {
        if polynomial_tail(i).is_some() {
            return polynomial_band(s, o, i);
        }
        return Ok(missing(
            r,
            "signed weighted atoms, band degree, coordinate and tail coverage required",
        ));
    };
    let p = o.working_precision_bits;
    let source_precision = i.unwrap().precision_bits;
    let point = |x: &str| -> Result<Float> { Ok(Float::with_val(p, scalar(x, source_precision)?)) };
    let d = b.degree;
    if b.atoms.len() < d {
        return Ok(unresolved(r, "fewer supplied atoms than band degree"));
    }
    let row_bytes = (b.atoms.len() as u64).saturating_mul(u64::from(p).div_ceil(8) + 64);
    let resident = row_bytes.saturating_mul(14).saturating_add(
        (d as u64)
            .saturating_pow(2)
            .saturating_mul(u64::from(p).div_ceil(8) + 64),
    );
    let limit = o.maximum_working_bytes.unwrap_or(8 << 30);
    if resident > limit {
        return Ok(unresolved(
            r,
            "band recurrence working memory budget exceeded",
        ));
    }
    let arithmetic_bytes = (limit - resident) / 2;
    let capacity =
        ((limit - resident - arithmetic_bytes) / row_bytes.max(1)).min(d as u64) as usize;
    let x = b
        .atoms
        .iter()
        .map(|a| point(&a.coordinate))
        .collect::<Result<Vec<_>>>()?;
    let w = b
        .atoms
        .iter()
        .map(|a| point(&a.signed_weight))
        .collect::<Result<Vec<_>>>()?;
    // Positive common weight scaling and coordinate scaling preserve model
    // roots. Keep the recurrence in these normalized units, then restore each
    // diagnostic's physical dimension. Even weight exponents avoid introducing
    // an unnecessary sqrt(2) for an otherwise exact power-of-two mass.
    use super::retained_evidence::finite_math::scale_float;
    let coordinate_exponent = x
        .iter()
        .filter_map(Float::get_exp)
        .max()
        .map_or(0, i64::from);
    let weight_exponent = w
        .iter()
        .filter_map(Float::get_exp)
        .max()
        .map_or(0, |e| 2 * i64::from(e.div_euclid(2)));
    let x = x
        .iter()
        .map(|x| scale_float(x, -coordinate_exponent, p))
        .collect::<Result<Vec<_>>>()?;
    let w = w
        .iter()
        .map(|w| scale_float(w, -weight_exponent, p))
        .collect::<Result<Vec<_>>>()?;
    let restore = |x: &Float, power: i64| scale_float(x, coordinate_exponent * power, p);
    let inner =
        |a: &[Float], b: &[Float]| super::band_runtime::inner(a, b, &w, p, arithmetic_bytes);
    let absolute_weights = w.iter().map(|x| x.clone().abs()).collect::<Vec<_>>();
    let one = vec![Float::with_val(p, 1); x.len()];
    let mass = inner(&one, &one)?;
    put(
        &mut r.values,
        "signed_mass",
        &scale_float(&mass, weight_exponent, p)?,
    );
    if mass <= 0 {
        return Ok(unresolved(
            r,
            "signed functional is not positive on constants",
        ));
    }
    let guard = Float::with_val(p, 1) >> (p / 2);
    if super::band_runtime::disk_estimate(x.len(), p, d) > super::band_runtime::disk_budget()? {
        return Ok(unresolved(
            r,
            "band basis disk estimate exceeds XC_RESEARCH_BASIS_BYTES; raise the explicit diagnostic disk budget",
        ));
    }
    let identity = (
        "signed-band-recurrence-v4-normalized-exact-contractions",
        coordinate_exponent,
        weight_exponent,
        source_precision,
        &s.manifest.content_digest,
        b,
        o,
    );
    let mut vectors = super::band_runtime::BandVectors::new(&identity, x.len(), p, capacity, d)?;
    #[derive(Serialize, Deserialize)]
    struct Progress {
        next: usize,
        alpha: Vec<String>,
        beta: Vec<String>,
        rows: Vec<AnalysisRow>,
    }
    let mut alpha = Vec::new();
    let mut beta = Vec::new();
    let mut begin = 0;
    if let Some(saved) = vectors.store.load::<Progress>("recurrence-progress")? {
        let count = saved.next.min(d.saturating_sub(1)) + 1;
        if saved.next <= d
            && saved.rows.len() == saved.next
            && saved.alpha.len() == saved.next
            && saved.beta.len() == saved.next.min(d - 1)
            && (0..count).all(|j| vectors.get(j).is_ok())
        {
            alpha = coeffs(&saved.alpha, p)?;
            beta = coeffs(&saved.beta, p)?;
            r.rows = saved.rows;
            begin = saved.next;
            xc_core::progress_message!("band recurrence resumed at degree {begin}/{d}");
        } else {
            vectors.clear_memory();
        }
    }
    if begin == 0 {
        let scale = mass.clone().sqrt();
        vectors.save(0, one.into_iter().map(|a| a / &scale).collect())?;
    }
    let mut jacobi = vec![Float::with_val(p, 0); d * d];
    for (j, a) in alpha.iter().enumerate() {
        jacobi[j * d + j] = a.clone();
    }
    for (j, b) in beta.iter().enumerate() {
        jacobi[j * d + j + 1] = b.clone();
        jacobi[(j + 1) * d + j] = b.clone();
    }
    for j in begin..d {
        let _stage = super::capture_runtime::Stage::new(format!("band recurrence {}/{d}", j + 1));
        let q = vectors.get(j)?;
        let xq = x
            .iter()
            .zip(&q)
            .map(|(x, q)| Float::with_val(p, x) * q)
            .collect::<Vec<_>>();
        let a = inner(&q, &xq)?;
        alpha.push(a.clone());
        jacobi[j * d + j] = a.clone();
        let mut v = xq
            .iter()
            .zip(&q)
            .map(|(x, q)| Float::with_val(p, x) - Float::with_val(p, &a) * q)
            .collect::<Vec<_>>();
        if j > 0 {
            super::band_runtime::subtract(&mut v, &vectors.get(j - 1)?, &beta[j - 1], p);
        }
        let mut leakage = Float::with_val(p, 0);
        for _ in 0..2 {
            for k in 0..=j {
                let q = vectors.get(k)?;
                let c = inner(&v, &q)?;
                leakage += c.clone().abs();
                super::band_runtime::subtract(&mut v, &q, &c, p);
            }
        }
        let norm = inner(&v, &v)?;
        let absolute = super::band_runtime::inner(&v, &v, &absolute_weights, p, arithmetic_bytes)?;
        let mut rr = row(j + 1, "signed_functional_stieltjes_recurrence");
        put(&mut rr.values, "jacobi_diagonal", &restore(&a, 1)?);
        for (name, value, power) in [
            ("reorthogonalization_correction", &leakage, 1),
            ("next_signed_norm_squared", &norm, 2),
            ("next_absolute_norm_squared", &absolute, 2),
        ] {
            match restore(value, power) {
                Ok(value) => put(&mut rr.values, name, &value),
                Err(error) => {
                    rr.outcome = "unresolved_denominator".into();
                    rr.notes.push(format!("{name} outside the representable output range; normalized recurrence retained: {error}"));
                }
            }
        }
        if j > 0 {
            put(
                &mut rr.values,
                "jacobi_off_diagonal_previous",
                &restore(&beta[j - 1], 1)?,
            );
        }
        if j + 1 < d {
            if norm <= Float::with_val(p, &absolute) * &guard {
                rr.outcome = "unresolved_denominator".into();
                r.rows.push(rr);
                r.outcome = "partial_unresolved".into();
                r.reason=Some("signed functional positivity or arithmetic resolution fails before requested degree; coverage and precision must be reviewed".into());
                return Ok(r);
            }
            let next = norm.sqrt();
            beta.push(next.clone());
            jacobi[j * d + j + 1] = next.clone();
            jacobi[(j + 1) * d + j] = next.clone();
            vectors.save(j + 1, v.into_iter().map(|v| v / &next).collect())?;
        }
        r.rows.push(rr);
        vectors.store.save(
            "recurrence-progress",
            &Progress {
                next: j + 1,
                alpha: alpha.iter().map(dec).collect(),
                beta: beta.iter().map(dec).collect(),
                rows: r.rows.clone(),
            },
        )?;
    }
    if r.rows.iter().any(|r| r.outcome != "point_measurement") {
        r.outcome = "partial_unresolved".into();
    }
    let mut eigenvalues = if let Some(values) = vectors.store.load::<Vec<String>>("jacobi-roots")? {
        coeffs(&values, p)?
    } else {
        let values = xc_numerics::eigen::dense_symmetric_eigenvalues_hp_stable(&jacobi, d, p)?;
        if let Err(e) = vectors
            .store
            .save("jacobi-roots", &values.iter().map(dec).collect::<Vec<_>>())
        {
            xc_core::progress_message!("band checkpoint unavailable: {e}");
        }
        values
    };
    eigenvalues = eigenvalues
        .iter()
        .map(|x| restore(x, 1))
        .collect::<Result<Vec<_>>>()?;
    eigenvalues.sort_by(Float::total_cmp);
    for (j, root) in eigenvalues.iter().enumerate() {
        put(&mut r.rows[j].values, "model_band_root", root);
        if let Some(other) = b.scoring_roots.get(j) {
            put(
                &mut r.rows[j].values,
                "scoring_root_difference",
                &(Float::with_val(p, root) - point(other)?),
            );
        }
    }
    match super::band_runtime::inverse_moments(&eigenvalues, p, arithmetic_bytes) {
        Ok(values) => {
            for (name, value) in ["one", "two", "three"].iter().zip(&values) {
                put(&mut r.values, &format!("band_inverse_moment_{name}"), value);
            }
        }
        Err(error) => {
            r.outcome = "partial_unresolved".into();
            r.rows[0].notes.push(format!(
                "band inverse moments unresolved; model roots retained: {error}"
            ));
        }
    }
    put(
        &mut r.values,
        "band_inverse_moment_root_count",
        &Float::with_val(p, eigenvalues.len()),
    );
    r.rows[0].notes.push(format!(
        "inverse moments include all model roots in coordinate {}; no omitted-root completion",
        b.coordinate
    ));
    if let Some(e) = &b.input_energy {
        put(&mut r.values, "borrowed_input_energy", &point(e)?);
    }
    r.reason = Some(format!(
        "coverage: {}; borrowed inputs: {}; hypotheses: {}",
        b.coverage,
        b.borrowed_inputs.join("; "),
        b.hypotheses.join("; ")
    ));
    r.convention="signed finite atomic functional; twice-reorthogonalized Stieltjes recurrence; Jacobi roots are model band roots, not generalized energy eigenvalues; declared ordinate coverage and energy inputs; no RH premise or infinite-tail certification".into();
    Ok(r)
}
fn polynomial_tail(
    i: Option<&ExternalResearchInputs>,
) -> Option<&super::convergence_capture::TailForm> {
    i?.run_once
        .as_ref()?
        .tail_form
        .as_ref()
        .filter(|f| f.polynomial_coordinate.is_some())
}

// Scale every exact binary coefficient by a common power of two using
// integers, then form a strict Cauchy radius exactly. A rounded floating ratio
// can shrink a search window; its overflow can also occur before finite roots
// are isolated. Common scaling keeps even huge equal-scale coefficients small.
#[cfg(feature = "arb")]
fn polynomial_root_window(
    coefficients: &[Float],
    p: u32,
    maximum_bytes: u64,
) -> Result<(Vec<rug::Rational>, rug::Rational)> {
    if !(64..=1_000_000).contains(&p)
        || !(2..=2049).contains(&coefficients.len())
        || coefficients.iter().any(|x| !x.is_finite())
        || coefficients.last().is_none_or(Float::is_zero)
    {
        bail!("invalid finite polynomial root window inputs");
    }
    let points = coefficients
        .iter()
        .map(|x| {
            x.to_integer_exp()
                .ok_or_else(|| anyhow::anyhow!("nonfinite polynomial coefficient"))
        })
        .collect::<Result<Vec<_>>>()?;
    let common = points
        .iter()
        .filter(|(m, _)| !m.is_zero())
        .map(|(_, e)| i64::from(*e))
        .min()
        .ok_or_else(|| anyhow::anyhow!("zero polynomial"))?;
    let span = points
        .iter()
        .filter(|(m, _)| !m.is_zero())
        .map(|(m, e)| (i64::from(*e) - common) as u64 + u64::from(m.significant_bits()))
        .max()
        .unwrap_or(0);
    if span > 8_000_000
        || (span.div_ceil(8) + 64)
            .saturating_mul(coefficients.len() as u64)
            .saturating_mul(16)
            > maximum_bytes
    {
        bail!("exact polynomial root window exceeds coefficient span or workspace budget");
    }
    let rational = points
        .into_iter()
        .map(|(mut mantissa, exponent)| -> Result<_> {
            if !mantissa.is_zero() {
                mantissa <<= u32::try_from(i64::from(exponent) - common)?;
            }
            Ok(rug::Rational::from(mantissa))
        })
        .collect::<Result<Vec<_>>>()?;
    let lead = rational.last().unwrap().clone().abs();
    let maximum = rational[..rational.len() - 1]
        .iter()
        .map(|x| x.clone().abs() / &lead)
        .max()
        .unwrap();
    // Cauchy gives |root| <= 1 + max |a_j/a_n|. Add a second unit so
    // the open FLINT isolation window strictly contains every real root.
    Ok((rational, maximum + 2u32))
}

/// Recover roots of the solved polynomial, not eigenvalues of its energy pencil.
#[cfg(feature = "arb")]
fn polynomial_band(
    s: &RetainedState,
    o: &ExtensionOptions,
    i: Option<&ExternalResearchInputs>,
) -> Result<ExtendedAnalysis> {
    let f = polynomial_tail(i).unwrap();
    let p = o.working_precision_bits;
    let model = super::convergence_capture::tail_operator(
        s,
        o,
        i.and_then(|i| i.run_once.as_ref()),
        i.unwrap().precision_bits,
    )?;
    let mut r = report("band_reconstruction", s, o);
    if let Some(energy) = model.values.get("model_energy") {
        let point = |x: &str| -> Result<Float> {
            Ok(Float::with_val(p, scalar(x, i.unwrap().precision_bits)?))
        };
        let mass = point(&f.finite_zero_form[0])? + point(&f.tail_correction[0])?
            - scalar(energy, p)? * point(&f.lattice_gram[0])? / 2u32;
        put(&mut r.values, "signed_mass", &mass);
    }
    let coefficients = (0..f.dimension)
        .map(|j| {
            model
                .values
                .get(&format!("model_vector_coefficient_{j}"))
                .map(|v| scalar(v, p))
                .transpose()
        })
        .collect::<Result<Option<Vec<_>>>>()?;
    let Some(coefficients) = coefficients else {
        return Ok(unresolved(
            r,
            "arithmetic model vector unavailable; see tail-operator diagnostic",
        ));
    };
    let lead = coefficients.last().unwrap().clone().abs();
    if lead == 0 {
        return Ok(unresolved(
            r,
            "model polynomial leading coefficient is zero",
        ));
    }
    let (rational, bound) =
        polynomial_root_window(&coefficients, p, o.maximum_working_bytes.unwrap_or(8 << 30))?;
    let (roots, square_free) =
        super::arb_bridge::rational_polynomial_real_roots(&rational, &(-bound.clone()), &bound, p)?;
    put(
        &mut r.values,
        "band_degree",
        &Float::with_val(p, f.dimension - 1),
    );
    put(
        &mut r.values,
        "band_inverse_moment_root_count",
        &Float::with_val(p, roots.len()),
    );
    for (j, root) in roots.iter().enumerate() {
        r.rows.push(polynomial_root_row(j + 1, root)?);
    }
    if roots.len() != f.dimension - 1 || !square_free || model.outcome != "point_measurement" {
        r.outcome = "partial_unresolved".into();
        r.reason = Some(
            "model vector qualification or full simple real polynomial root count unresolved"
                .into(),
        );
    }
    let moments = if roots.len() == f.dimension - 1 && square_free {
        super::band_runtime::polynomial_inverse_moments(
            &coefficients,
            p,
            o.maximum_working_bytes.unwrap_or(8 << 30),
        )
    } else {
        Err(anyhow::anyhow!(
            "complete simple real polynomial root coverage unavailable"
        ))
    };
    match moments {
        Ok(values) => {
            for (label, value) in ["one", "two", "three"].iter().zip(&values) {
                put(
                    &mut r.values,
                    &format!("band_inverse_moment_{label}"),
                    value,
                );
            }
        }
        Err(error) => {
            r.outcome = "partial_unresolved".into();
            let note = format!("polynomial inverse moments withheld: {error}");
            r.reason = Some(match r.reason.take() {
                Some(reason) => format!("{reason}; {note}"),
                None => note,
            });
        }
    }
    r.convention = format!(
        "arithmetic-completed fixed-prefix model in {}; solve (head+remainder)*b=(E/2)*Gram*b then roots of sum b_j*z^j; no retained energy, eigenvector, or band roots used for the solve; intervals enclose only the rounded model polynomial, not actual CCM roots; {}",
        f.polynomial_coordinate.as_ref().unwrap(),
        f.coverage
    );
    Ok(r)
}
#[cfg(feature = "arb")]
fn polynomial_root_row(
    ordinal: usize,
    root: &xc_numerics::mpfr_interval::MpfrInterval,
) -> Result<AnalysisRow> {
    root.validate()?;
    let mut r = row(ordinal, "arithmetic_model_polynomial_root");
    put(
        &mut r.values,
        "model_band_root",
        root.midpoint_point().lower(),
    );
    let digits = Some((u64::from(root.precision()) * 30103 / 100000 + 10) as usize);
    r.values.insert(
        "rounded_polynomial_root_lower".into(),
        root.lower()
            .to_string_radix_round(10, digits, rug::float::Round::Down),
    );
    r.values.insert(
        "rounded_polynomial_root_upper".into(),
        root.upper()
            .to_string_radix_round(10, digits, rug::float::Round::Up),
    );
    Ok(r)
}

#[cfg(not(feature = "arb"))]
fn polynomial_band(
    s: &RetainedState,
    o: &ExtensionOptions,
    _i: Option<&ExternalResearchInputs>,
) -> Result<ExtendedAnalysis> {
    Ok(unresolved(
        report("band_reconstruction", s, o),
        "polynomial model root isolation requires the arb feature",
    ))
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn analyze(
    id: &str,
    s: &RetainedState,
    m: Option<&RetainedMatrix<'_>>,
    roots: Option<&RetainedRoots>,
    o: &ExtensionOptions,
    i: Option<&ExternalResearchInputs>,
) -> Result<ExtendedAnalysis> {
    match id {
        "capture_preflight" => preflight(s, m, roots, o, i),
        "consistency" => consistency_math::analyze(s, o, i),
        "configuration_comparison" => comparisons(s, m, roots, o, i),
        "band_reconstruction" => super::atom_research::band_report(s, o, i),
        "transform_enclosure" => super::transform_enclosure::analyze(
            s,
            roots,
            o,
            completion(i),
            i.map_or(s.precision, |i| i.precision_bits),
        ),
        _ => bail!("unknown completion diagnostic"),
    }
}

/// Assemble an explicit finite polynomial-basis energy model from declared
/// zero/lattice weights. The caller supplies every tail correction or explicitly
/// chooses a finite-only model. No RH or missing-ordinate assumption is made.
pub fn prepare_tail_form(
    definition_digest: ContentDigest,
    basis_polynomials: &[Vec<String>],
    atoms: &[WeightedAtom],
    tail_correction: Option<&[String]>,
    coverage: &str,
    hypotheses: &[String],
    precision_bits: u32,
) -> Result<super::convergence_capture::TailForm> {
    prepare_tail_form_at_precision(
        definition_digest,
        basis_polynomials,
        atoms,
        tail_correction,
        coverage,
        hypotheses,
        precision_bits,
        precision_bits,
        super::capture_runtime::CaptureResourcePolicy::from_environment()?.maximum_working_bytes,
    )
}

/// Decode the recipe at its owning precision before exact polynomial arithmetic.
/// The resulting form entries are rounded once at the requested working precision.
#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_tail_form_at_precision(
    definition_digest: ContentDigest,
    basis_polynomials: &[Vec<String>],
    atoms: &[WeightedAtom],
    tail_correction: Option<&[String]>,
    coverage: &str,
    hypotheses: &[String],
    source_precision: u32,
    p: u32,
    maximum_bytes: u64,
) -> Result<super::convergence_capture::TailForm> {
    precision(source_precision)?;
    precision(p)?;
    let n = basis_polynomials.len();
    if source_precision > p
        || n == 0
        || n > 128
        || coverage.is_empty()
        || !definition_digest.validate()
        || basis_polynomials
            .iter()
            .any(|v| v.is_empty() || v.len() > 2049)
    {
        bail!("invalid finite tail-form recipe");
    }
    let points = (basis_polynomials.iter().map(Vec::len).sum::<usize>() as u64)
        .saturating_add((atoms.len() as u64).saturating_mul(2))
        .saturating_add((n as u64).saturating_pow(2).saturating_mul(4))
        .saturating_add(64);
    if points
        .saturating_mul(u64::from(p).div_ceil(8) + 128)
        .saturating_mul(3)
        > maximum_bytes
    {
        bail!("tail-form preparation workspace budget exceeded");
    }
    let point = |x: &str| -> Result<Float> { Ok(Float::with_val(p, scalar(x, source_precision)?)) };
    let basis = basis_polynomials
        .iter()
        .map(|v| v.iter().map(|x| point(x)).collect::<Result<Vec<_>>>())
        .collect::<Result<Vec<_>>>()?;
    let atoms = atoms
        .iter()
        .map(|a| -> Result<_> {
            let zero = match a.family.as_str() {
                "zero" => true,
                "lattice" => false,
                _ => bail!("unknown weighted atom family"),
            };
            Ok((point(&a.coordinate)?, point(&a.weight)?, zero))
        })
        .collect::<Result<Vec<_>>>()?;
    let (zero, lattice) = tail_form_math::forms(&basis, &atoms, p, maximum_bytes)?;
    let tail = if let Some(v) = tail_correction {
        if v.len() != n * n {
            bail!("tail correction shape mismatch");
        }
        v.iter().map(|x| point(x)).collect::<Result<Vec<_>>>()?
    } else {
        vec![Float::with_val(p, 0); n * n]
    };
    for row in 0..n {
        for col in 0..row {
            if tail[row * n + col] != tail[col * n + row] {
                bail!("tail correction is not symmetric");
            }
        }
    }
    let mut hypotheses = hypotheses.to_vec();
    if tail_correction.is_none() {
        hypotheses.push(
            "finite supplied atoms only; omitted infinite tail is unknown, not proved zero".into(),
        );
    }
    Ok(super::convergence_capture::TailForm {
        polynomial_coordinate: None,
        definition_digest,
        dimension: n,
        finite_zero_form: zero.iter().map(dec).collect(),
        tail_correction: tail.iter().map(dec).collect(),
        lattice_gram: lattice.iter().map(dec).collect(),
        tail_operator_error: None,
        coverage: coverage.into(),
        hypotheses,
    })
}

// Both sides must keep their original stored points when their shared precision
// rises. Exact cutoff strings and independently tagged inputs retain ownership.
pub(super) fn promote_input_precision(
    input: &mut ExternalResearchInputs,
    target: u32,
) -> Result<()> {
    precision(input.precision_bits)?;
    precision(target)?;
    anyhow::ensure!(
        target >= input.precision_bits,
        "input precision promotion cannot discard stored bits"
    );
    if target != input.precision_bits {
        preparation_points::Promotion {
            source: input.precision_bits,
            target,
        }
        .bundle(input)?;
        input.precision_bits = target;
    }
    Ok(())
}

pub(super) fn align_input_precisions(
    left: &mut ExternalResearchInputs,
    right: &mut ExternalResearchInputs,
) -> Result<()> {
    precision(left.precision_bits)?;
    precision(right.precision_bits)?;
    let target = left.precision_bits.max(right.precision_bits);
    for input in [left, right] {
        promote_input_precision(input, target)?;
    }
    Ok(())
}

#[cfg(test)]
mod exhaustive_resumed_promotion_contract {
    use super::*;
    use serde_json::json;
    const ALIAS: &str = "1.0000000000000000000000000000000000000001";
    fn bundle(a: &str, p: u32) -> ExternalResearchInputs {
        let digest = ContentDigest::sha256(b"precision merge fixture");
        let run_once = json!({
          "component_actions":[{"label":"part","source_digest":digest,"action":[a,a,a],"convention":ALIAS}],"derivative_actions":[{"label":"derivative","source_digest":digest,"action":[a,a,a],"convention":ALIAS}],"log_cutoff_velocity":a,"reference_vectors":[[a,a,a]],
          "comparison":{"source_digest":digest,"matrix_digest":digest,"lambda_squared":"9","n_modes":0,"precision_bits":64,"coefficients":[ALIAS],"matrix":[ALIAS],"eigenvalue":ALIAS,"assembly_policy":ALIAS},
          "tail_form":{"definition_digest":digest,"dimension":1,"finite_zero_form":[a],"tail_correction":[a],"lattice_gram":[a],"tail_operator_error":a,"coverage":ALIAS,"hypotheses":[ALIAS]},
          "uncertainty":{"unit_state_l2_error":a,"source_certificate_digest":digest,"hypotheses":[ALIAS]},"producer_notes":[ALIAS],
          "completion":{
            "atom_analysis":{"maximum_atoms":10,"maximum_input_bytes":100000,"evaluations":[{"ordinal":1,"coordinate":a,"label":ALIAS}],"cutoffs":[a],"tail_recipe":{"basis_polynomials":[[a]],"tail_correction":[a],"hypotheses":[ALIAS]}},
            "independent_actions":[{"label":"direct","source_digest":digest,"action":[a,a,a],"convention":ALIAS}],
            "response_checks":[{"ordinal":1,"t":a,"source_digest":digest,"branch":ALIAS,"coordinate":"t","derivative_parameter":"s","activation_convention":ALIAS,"fixed_velocity":a,"support_velocity":a,"total_velocity":a}],
            "band":{"degree":1,"coordinate":"z","definition_digest":digest,"atoms":[{"coordinate":a,"signed_weight":a,"family":"zero"}],"coverage":ALIAS,"hypotheses":[ALIAS],"borrowed_inputs":[],"input_energy":a,"scoring_roots":[a]},
            "contour":{"left":"-1","right":a,"bottom":"-1","top":a,"maximum_depth":8,"maximum_segments":32}
          }
        });
        serde_json::from_value(json!({"schema_version":1,"source_eigenpair":digest,"lambda_squared":"9","n_modes":1,"precision_bits":p,"convention_id":ALIAS,"definition_digest":digest,"approximation_scope":ALIAS,
          "target":{"definition_digest":digest,"evaluation_policy":ALIAS,"approximation_scope":ALIAS,"intervals":8,"values":vec![a;9],"basis_values":[vec![a;9],vec![a;9]],"fixed_second_component":a,"raw_normalizer":a,"trial_coefficients":[a,a,a]},
          "reference_jets":[{"ordinal":1,"t":a,"reference_window":{"value":a,"derivative":a},"reference_full":{"value":a,"derivative":a},"exterior_tail":{"value":a,"derivative":a},"endpoint_tail_part":{"value":a,"derivative":a},"fitted_interior_parts":[{"value":a,"derivative":a}],"error_normalization":"unit_l2_dx","source_value_error":a,"tail_value_error":a,"source_derivative_error":a,"root_separation_radius":a}],
          "components":[{"label":"part","source_digest":digest,"diagonal":[a,a,a],"dense":vec![a;9],"rank_one":[{"weight":a,"vector":[a,a,a]}]}],
          "perturbations":[{"label":"derivative","source_digest":digest,"diagonal":[a,a,a],"dense":[],"rank_one":[{"weight":a,"vector":[a,a,a]}]}],
          "deficit":a,"deficit_kind":"exact_source",
          "atoms":[{"ordinal":1,"coordinate":a,"weight":a,"family":"zero","partition":ALIAS}],"atom_coordinate":ALIAS,"atom_coverage":ALIAS,"tail_checkpoints":[a],
          "cluster":[{"source_digest":digest,"n_modes":0,"precision_bits":64,"eigenvalue":ALIAS,"coefficients":[ALIAS],"assembly_policy":ALIAS}],
          "previous_cluster":[{"source_digest":digest,"n_modes":0,"precision_bits":64,"eigenvalue":ALIAS,"coefficients":[ALIAS],"assembly_policy":ALIAS}],"cluster_boundary_eigenvalues":[a,a],
          "energy_allowance":{"upper_trial_energy":a,"low_block_lower_bound":a,"high_block_lower_bound":a,"cross_block_norm_bound":a,"hypothesis_record_digest":digest,"hypotheses":[ALIAS]},
          "run_once":run_once
        })).unwrap()
    }
    fn check_and_remove_directed_premise_differences(
        alias: &mut ExternalResearchInputs,
        baseline: &ExternalResearchInputs,
    ) {
        use rug::{ops::Pow, Integer, Rational};
        let numerator = Integer::from_str_radix(&ALIAS.replace('.', ""), 10).unwrap();
        let exact = Rational::from((numerator, Integer::from(10).pow((ALIAS.len() - 2) as u32)));
        let ar = alias.run_once.as_mut().unwrap();
        let br = baseline.run_once.as_ref().unwrap();
        for (bound, original) in [
            (
                ar.tail_form
                    .as_mut()
                    .unwrap()
                    .tail_operator_error
                    .as_mut()
                    .unwrap(),
                br.tail_form
                    .as_ref()
                    .unwrap()
                    .tail_operator_error
                    .as_ref()
                    .unwrap(),
            ),
            (
                &mut ar.uncertainty.as_mut().unwrap().unit_state_l2_error,
                &br.uncertainty.as_ref().unwrap().unit_state_l2_error,
            ),
        ] {
            let upper = scalar(bound, alias.precision_bits).unwrap();
            assert!(
                upper >= exact && upper > 1,
                "non-dyadic declared upper bound was rounded inward"
            );
            assert_eq!(scalar(original, baseline.precision_bits).unwrap(), 1);
            *bound = original.clone();
        }
    }
    #[test]
    fn exhaustive_resumed_live_merge_preserves_existing_owned_points() {
        let mut baseline = bundle("1", 128);
        let mut alias = bundle(ALIAS, 128);
        baseline.validate().unwrap();
        alias.validate().unwrap();
        let mut first = bundle("2", 256);
        let mut second = first.clone();
        align_input_precisions(&mut baseline, &mut first).unwrap();
        align_input_precisions(&mut alias, &mut second).unwrap();
        check_and_remove_directed_premise_differences(&mut alias, &baseline);
        assert_eq!(
            alias, baseline,
            "raising bundle precision must preserve every inherited binary point and leave metadata/independently tagged points unchanged"
        );
        assert_eq!(alias.precision_bits, 256);
    }
    #[test]
    fn exhaustive_resumed_live_merge_preserves_incoming_owned_points() {
        let mut baseline = bundle("1", 128);
        let mut alias = bundle(ALIAS, 128);
        let mut first = bundle("2", 256);
        let mut second = first.clone();
        align_input_precisions(&mut first, &mut baseline).unwrap();
        align_input_precisions(&mut second, &mut alias).unwrap();
        assert_eq!(
            alias.precision_bits, 256,
            "incoming data must be serialized at the receiving precision before merging"
        );
        check_and_remove_directed_premise_differences(&mut alias, &baseline);
        assert_eq!(
            alias, baseline,
            "the incoming side also needs source-point promotion"
        );
    }
}

#[cfg(all(test, feature = "arb"))]
mod exhaustive_resumed_polynomial_window {
    use super::*;
    #[test]
    fn exhaustive_resumed_polynomial_window_preserves_extreme_common_scale() {
        let p = 128;
        for shift in [rug::float::exp_max() - 4, rug::float::exp_min() + 4] {
            let a = Float::with_val(p, 6) << shift;
            let b = Float::with_val(p, -3) << shift;
            let (coefficients, bound) = polynomial_root_window(&[a, b], p, 1 << 20).unwrap();
            assert_eq!(-coefficients[0].clone() / &coefficients[1], 2);
            assert_eq!(bound, 4);
        }
    }
    #[test]
    fn exhaustive_resumed_polynomial_window_rejects_unbounded_span_and_workspace() {
        let p = 128;
        assert!(polynomial_root_window(
            &[Float::with_val(p, 1), Float::with_val(p, 1) << 8_000_001u32],
            p,
            8 << 30
        )
        .is_err());
        assert!(
            polynomial_root_window(&[Float::with_val(p, 1), Float::with_val(p, 1)], p, 1).is_err()
        );
        assert!(polynomial_root_window(
            &[Float::with_val(p, 1), Float::with_val(p, 0)],
            p,
            8 << 30
        )
        .is_err());
    }
    #[test]
    fn exhaustive_resumed_polynomial_window_contains_exact_linear_root() {
        for p in [64, 128, 256] {
            for denominator in 3..32 {
                let a = Float::with_val(p, 1) << (p + 128);
                let b = Float::with_val(p, -denominator);
                let exact_root = -a.to_rational().unwrap() / b.to_rational().unwrap();
                let (_, bound) = polynomial_root_window(&[a, b], p, 8 << 30).unwrap();
                assert!(
                    bound > exact_root,
                    "floating Cauchy window excludes exact linear root at p={p}, denominator={denominator}"
                );
            }
        }
    }
}
