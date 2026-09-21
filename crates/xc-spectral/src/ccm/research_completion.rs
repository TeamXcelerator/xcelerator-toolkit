//! Reproducible preparation, comparisons and finite signed-measure models.
//! Every external convention and borrowed input stays in the source artifact.
use super::{
    convergence_capture::OperatorAction, extended_research::*, retained_evidence::*,
    state_geometry::RetainedState,
};
use anyhow::{bail, Result};
use rug::{ops::Pow, Float};
use serde::{Deserialize, Serialize};
use xc_cache::ContentDigest;
use xc_numerics::prefix::lossless_decimal as dec;

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
                || scalar(&a.lambda_squared, p)? <= 1
            {
                bail!("invalid comparison source shape");
            }
            for v in a.coefficients.iter().chain(std::iter::once(&a.eigenvalue)) {
                scalar(v, p)?;
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
                    scalar(v, p)?;
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
        let p = self.precision_bits.max(state.precision).saturating_add(64);
        precision(p)?;
        let mut input: ExternalResearchInputs = serde_json::from_value(
            serde_json::json!({"schema_version":1,"source_eigenpair":state.manifest.content_digest,"lambda_squared":state.cutoff,"n_modes":state.modes,"precision_bits":p,"convention_id":"external_reference_preparation_v1","definition_digest":self.definition_digest,"approximation_scope":self.approximation_scope}),
        )?;
        input.target = self.sampled_reference.clone();
        input.atoms = self.weighted_atoms.clone();
        input.atom_coordinate = self.atom_coordinate.clone();
        input.atom_coverage = self.atom_coverage.clone();
        input.run_once = Some(super::convergence_capture::RunOnceInputs {
            tail_form: self.tail_form.clone(),
            completion: self.completion.clone(),
            ..Default::default()
        });
        if let Some(recipe) = &self.tail_recipe {
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
                &self.weighted_atoms,
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
            let l = scalar(&state.cutoff, p)?.ln();
            let coefficients = coeffs(&f.reference.coefficients, p)?;
            if coefficients
                .iter()
                .zip(coefficients.iter().rev())
                .any(|(a, b)| a != b)
            {
                bail!("automatic real-even reference preparation requires even coefficients");
            }
            let center = super::retained_evidence::center(&coefficients, p);
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
                        coeffs(&b.coefficients, p)
                    })
                    .collect::<Result<Vec<_>>>()?;
                for j in 0..=intervals {
                    let x = Float::with_val(p, &l) * j / (2 * intervals);
                    values.push(dec(&super::extended_research::evaluate(
                        &normalized,
                        &x,
                        &l,
                        p,
                    )
                    .0));
                    for (v, out) in basis.iter().zip(&mut basis_values) {
                        out.push(dec(&super::extended_research::evaluate(v, &x, &l, p).0));
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
                // This is the transform of the explicitly finite reference, whose
                // exterior is zero by definition, not an omitted infinite target tail.
                for point in &roots.dataset.points {
                    let Some(t) = &point.value else { continue };
                    let t_value = scalar(t, p)?;
                    let mut reference_state = state.clone();
                    reference_state.coefficients = normalized.clone();
                    reference_state.modes = normalized.len() / 2;
                    let (value, derivative, _, _) = transform_terms(&reference_state, &t_value, p)?;
                    let scale = norm2(&normalized, p).sqrt() * l.clone().sqrt();
                    let jet = Jet {
                        value: dec(&(value * &scale)),
                        derivative: dec(&(derivative * &scale)),
                    };
                    input.reference_jets.push(ReferenceJet {
                        matched_root_ordinal: None,
                        ordinal: point.ordinal,
                        t: t.clone(),
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
fn consistency(
    s: &RetainedState,
    o: &ExtensionOptions,
    i: Option<&ExternalResearchInputs>,
) -> Result<ExtendedAnalysis> {
    let mut r = report("consistency", s, o);
    let Some(c) = completion(i) else {
        return Ok(missing(r, "independent component actions required"));
    };
    let p = o.working_precision_bits;
    let original = i
        .and_then(|i| i.run_once.as_ref())
        .map(|i| i.component_actions.as_slice())
        .unwrap_or(&[]);
    let v = source_unit(s, p);
    for action in &c.independent_actions {
        let mut rr = row(r.rows.len() + 1, action.label.clone());
        let reconstructed = original
            .iter()
            .find(|a| a.label == "tau_prime_reconstructed" && action.label == "tau_prime_direct");
        if let Some(other) = reconstructed {
            let delta = coeffs(&action.action, p)?
                .iter()
                .zip(coeffs(&other.action, p)?)
                .map(|(a, b)| Float::with_val(p, a) - b)
                .collect::<Vec<_>>();
            put(
                &mut rr.values,
                "action_difference_norm",
                &norm2(&delta, p).sqrt(),
            );
            put(
                &mut rr.values,
                "signed_energy_difference",
                &dot(&v, &delta, p),
            );
            rr.notes.push(format!(
                "independent source {}; reconstructed source {}; {}",
                action.source_digest.0, other.source_digest.0, action.convention
            ));
        } else {
            rr.outcome = "missing_input".into();
            rr.notes
                .push("no convention-compatible reconstructed action".into());
        }
        r.rows.push(rr);
    }
    if r.rows.is_empty() {
        return Ok(missing(r,"direct retained prime component unavailable; algebraic closure is not independent validation"));
    }
    r.convention="direct retained component contraction versus algebraic reconstruction on the same signed unit state; signed differences, no automatic theorem".into();
    Ok(r)
}
fn comparisons(
    s: &RetainedState,
    m: Option<&RetainedMatrix<'_>>,
    roots: Option<&RetainedRoots>,
    o: &ExtensionOptions,
    i: Option<&ExternalResearchInputs>,
) -> Result<ExtendedAnalysis> {
    let mut r = report("configuration_comparison", s, o);
    let Some(c) = completion(i).filter(|c| !c.comparisons.is_empty()) else {
        return Ok(missing(
            r,
            "comparison snapshots or a verified retained-source cohort required",
        ));
    };
    let p = o.working_precision_bits;
    let mut seen = std::collections::BTreeSet::new();
    for (idx, c) in c.comparisons.iter().enumerate() {
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
        if !seen.insert((
            &a.lambda_squared,
            a.n_modes,
            a.precision_bits,
            &c.assembly_policy,
            &c.quadrature_policy,
            &c.selection_policy,
        )) {
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
        for (name, value) in [
            ("comparison_C", scalar(&a.lambda_squared, p)?),
            ("comparison_N", Float::with_val(p, a.n_modes)),
            ("comparison_P", Float::with_val(p, a.precision_bits)),
            (
                "signed_energy_difference",
                scalar(&s.eigenvalue, p)? - scalar(&a.eigenvalue, p)?,
            ),
        ] {
            put(&mut rr.values, name, &value);
        }
        if a.lambda_squared == s.cutoff && a.n_modes <= s.modes {
            let raw = coeffs(&a.coefficients, p)?;
            let norm = norm2(&raw, p).sqrt();
            if norm == 0 {
                rr.outcome = "unresolved_denominator".into();
                r.rows.push(rr);
                continue;
            }
            let mut embedded = vec![Float::with_val(p, 0); s.coefficients.len()];
            let offset = s.modes - a.n_modes;
            for (j, x) in raw.iter().enumerate() {
                embedded[offset + j] = Float::with_val(p, x) / &norm;
            }
            let overlap = dot(&source_unit(s, p), &embedded, p);
            put(
                &mut rr.values,
                "absolute_unit_overlap",
                &overlap.clone().abs(),
            );
            if let Some(m) = m {
                let action = matvec(m, &embedded, p);
                let e = scalar(&a.eigenvalue, p)?;
                let mut low = Float::with_val(p, 0);
                let mut high = Float::with_val(p, 0);
                for (j, x) in action.iter().enumerate() {
                    if (offset..offset + raw.len()).contains(&j) {
                        low += (Float::with_val(p, x) - Float::with_val(p, &e) * &embedded[j])
                            .square();
                    } else {
                        high += x.clone().square();
                    }
                }
                put(
                    &mut rr.values,
                    "independent_small_state_low_residual_squared",
                    &low,
                );
                put(
                    &mut rr.values,
                    "independent_small_state_high_forcing_squared",
                    &high,
                );
            }
        } else {
            rr.notes.push(
                "different supports or larger N: no same-basis coefficient or matrix subtraction"
                    .into(),
            );
        }
        let branch = roots
            .map(|r| serde_json::to_string(&r.acquisition))
            .transpose()?
            .unwrap_or_default();
        if c.root_coordinate == "mellin_t" && c.root_branch == branch {
            if let Some(roots) = roots {
                for point in &roots.dataset.points {
                    if let (Some(t), Some(other)) = (
                        &point.value,
                        c.roots
                            .iter()
                            .find(|r| r.ordinal == point.ordinal)
                            .and_then(|r| r.value.as_ref()),
                    ) {
                        put(
                            &mut rr.values,
                            &format!("root_{}_signed_difference", point.ordinal),
                            &(scalar(t, p)? - scalar(other, p)?),
                        );
                        let (v, d, _, _) = transform_terms(s, &scalar(t, p)?, p)?;
                        let mut other_state = s.clone();
                        other_state.cutoff = a.lambda_squared.clone();
                        other_state.modes = a.n_modes;
                        other_state.coefficients = coeffs(&a.coefficients, p)?;
                        let (ov, od, _, _) = transform_terms(&other_state, &scalar(t, p)?, p)?;
                        for (label, x, y) in [("value", v, ov), ("slope", d, od)] {
                            put(
                                &mut rr.values,
                                &format!("root_{}_transform_{label}_difference", point.ordinal),
                                &(x - y),
                            );
                        }
                    }
                }
            }
        } else {
            rr.notes.push(
                "root branch/coordinate mismatch or absence: ordinal differences withheld".into(),
            );
        }
        r.rows.push(rr);
    }
    r.convention="explicit independent C/N/P/quadrature cohorts; no fitted rate and no substituted parent prefix; root joins require equal acquisition branch and ordinal".into();
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
    let capacity = ((limit - resident) / row_bytes.max(1)).min(d as u64) as usize;
    let x = b
        .atoms
        .iter()
        .map(|a| scalar(&a.coordinate, p))
        .collect::<Result<Vec<_>>>()?;
    let w = b
        .atoms
        .iter()
        .map(|a| scalar(&a.signed_weight, p))
        .collect::<Result<Vec<_>>>()?;
    let inner = |a: &[Float], b: &[Float]| super::band_runtime::inner(a, b, &w, p);
    let absolute_weights = w.iter().map(|x| x.clone().abs()).collect::<Vec<_>>();
    let one = vec![Float::with_val(p, 1); x.len()];
    let mass = inner(&one, &one);
    put(&mut r.values, "signed_mass", &mass);
    if mass <= 0 {
        return Ok(unresolved(
            r,
            "signed functional is not positive on constants",
        ));
    }
    let guard = Float::with_val(p, 1) >> (p / 2);
    if super::band_runtime::disk_estimate(x.len(), p, d) > super::band_runtime::disk_budget()? {
        return Ok(unresolved(r,"band basis disk estimate exceeds XC_RESEARCH_BASIS_BYTES; raise the explicit diagnostic disk budget"));
    }
    let identity = (
        "signed-band-recurrence-v2",
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
            eprintln!("band recurrence resumed at degree {begin}/{d}");
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
        let a = inner(&q, &xq);
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
                let c = inner(&v, &q);
                leakage += c.clone().abs();
                super::band_runtime::subtract(&mut v, &q, &c, p);
            }
        }
        let norm = inner(&v, &v);
        let absolute = super::band_runtime::inner(&v, &v, &absolute_weights, p);
        let mut rr = row(j + 1, "signed_functional_stieltjes_recurrence");
        put(&mut rr.values, "jacobi_diagonal", &a);
        put(&mut rr.values, "reorthogonalization_correction", &leakage);
        put(&mut rr.values, "next_signed_norm_squared", &norm);
        put(&mut rr.values, "next_absolute_norm_squared", &absolute);
        if j > 0 {
            put(&mut rr.values, "jacobi_off_diagonal_previous", &beta[j - 1]);
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
    let mut eigenvalues = if let Some(values) = vectors.store.load::<Vec<String>>("jacobi-roots")? {
        coeffs(&values, p)?
    } else {
        let values = xc_numerics::eigen::dense_symmetric_eigenvalues_hp_stable(&jacobi, d, p)?;
        if let Err(e) = vectors
            .store
            .save("jacobi-roots", &values.iter().map(dec).collect::<Vec<_>>())
        {
            eprintln!("band checkpoint unavailable: {e}");
        }
        values
    };
    eigenvalues.sort_by(Float::total_cmp);
    for (j, root) in eigenvalues.iter().enumerate() {
        put(&mut r.rows[j].values, "model_band_root", root);
        if let Some(other) = b.scoring_roots.get(j) {
            put(
                &mut r.rows[j].values,
                "scoring_root_difference",
                &(Float::with_val(p, root) - scalar(other, p)?),
            );
        }
    }
    let zero_guard = (eigenvalues
        .iter()
        .map(|x| x.clone().abs())
        .max_by(Float::total_cmp)
        .unwrap()
        + 1u32)
        >> (p - 32);
    let inverse_resolved = eigenvalues.iter().all(|x| x.clone().abs() > zero_guard);
    if !inverse_resolved {
        r.outcome = "partial_unresolved".into();
    }
    for (power, label) in [(1, "one"), (2, "two"), (3, "three")] {
        if inverse_resolved {
            let sum = eigenvalues.iter().fold(Float::with_val(p, 0), |sum, x| {
                sum + (Float::with_val(p, 1) / x).pow(power)
            });
            put(&mut r.values, &format!("band_inverse_moment_{label}"), &sum);
        } else {
            r.rows[0].notes.push(format!(
                "band inverse moment {power} unresolved: a model root is zero or unresolved at working precision"
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
        put(&mut r.values, "borrowed_input_energy", &scalar(e, p)?);
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

/// Recover roots of the solved polynomial, not eigenvalues of its energy pencil.
#[cfg(feature = "arb")]
fn polynomial_band(
    s: &RetainedState,
    o: &ExtensionOptions,
    i: Option<&ExternalResearchInputs>,
) -> Result<ExtendedAnalysis> {
    let f = polynomial_tail(i).unwrap();
    let p = o.working_precision_bits;
    let model =
        super::convergence_capture::tail_operator(s, o, i.and_then(|i| i.run_once.as_ref()))?;
    let mut r = report("band_reconstruction", s, o);
    if let Some(energy) = model.values.get("model_energy") {
        let mass = scalar(&f.finite_zero_form[0], p)? + scalar(&f.tail_correction[0], p)?
            - scalar(energy, p)? * scalar(&f.lattice_gram[0], p)? / 2u32;
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
    let bound = coefficients
        .iter()
        .map(|x| x.clone().abs() / &lead)
        .max_by(Float::total_cmp)
        .unwrap()
        + 2u32;
    let rational = coefficients
        .iter()
        .map(|x| x.to_rational().unwrap())
        .collect::<Vec<_>>();
    let (roots, square_free) = super::arb_bridge::rational_polynomial_real_roots(
        &rational,
        &(-bound.clone()).to_rational().unwrap(),
        &bound.to_rational().unwrap(),
        p,
    )?;
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
    let mut moments = vec![Float::with_val(p, 0); 3];
    let mut inverse_defined = true;
    for (j, root) in roots.iter().enumerate() {
        let value = (Float::with_val(p, root.lower()) + root.upper()) / 2u32;
        let mut rr = row(j + 1, "arithmetic_model_polynomial_root");
        put(&mut rr.values, "model_band_root", &value);
        put(
            &mut rr.values,
            "rounded_polynomial_root_lower",
            root.lower(),
        );
        put(
            &mut rr.values,
            "rounded_polynomial_root_upper",
            root.upper(),
        );
        if root.lower() <= &0 && root.upper() >= &0 {
            inverse_defined = false;
        } else {
            let inverse = Float::with_val(p, 1) / &value;
            let mut term = inverse.clone();
            for m in &mut moments {
                *m += &term;
                term *= &inverse;
            }
        }
        r.rows.push(rr);
    }
    if roots.len() != f.dimension - 1 || !square_free || model.outcome != "point_measurement" {
        r.outcome = "partial_unresolved".into();
        r.reason = Some(
            "model vector qualification or full simple real polynomial root count unresolved"
                .into(),
        );
    }
    if inverse_defined && roots.len() == f.dimension - 1 {
        for (label, value) in ["one", "two", "three"].iter().zip(&moments) {
            put(
                &mut r.values,
                &format!("band_inverse_moment_{label}"),
                value,
            );
        }
    }
    r.convention=format!("arithmetic-completed fixed-prefix model in {}; solve (head+remainder)*b=(E/2)*Gram*b then roots of sum b_j*z^j; no retained energy, eigenvector, or band roots used for the solve; intervals enclose only the rounded model polynomial, not actual CCM roots; {}",f.polynomial_coordinate.as_ref().unwrap(),f.coverage);
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
        "consistency" => consistency(s, o, i),
        "configuration_comparison" => comparisons(s, m, roots, o, i),
        "band_reconstruction" => super::atom_research::band_report(s, o, i),
        "transform_enclosure" => super::transform_enclosure::analyze(s, roots, o, completion(i)),
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
    precision(precision_bits)?;
    let p = precision_bits;
    let n = basis_polynomials.len();
    if n == 0
        || n > 128
        || coverage.is_empty()
        || !definition_digest.validate()
        || basis_polynomials
            .iter()
            .any(|b| b.is_empty() || b.len() > 2049)
    {
        bail!("invalid finite tail-form recipe");
    }
    let memory = (n as u64)
        .saturating_mul(n as u64)
        .saturating_mul(u64::from(p).div_ceil(8) + 64)
        .saturating_mul(6);
    if memory
        > super::capture_runtime::CaptureResourcePolicy::from_environment()?.maximum_working_bytes
    {
        bail!("tail-form preparation workspace budget exceeded");
    }
    let basis = basis_polynomials
        .iter()
        .map(|v| coeffs(v, p))
        .collect::<Result<Vec<_>>>()?;
    let mut zero = vec![Float::with_val(p, 0); n * n];
    let mut lattice = zero.clone();
    for a in atoms {
        let x = scalar(&a.coordinate, p)?;
        let weight = scalar(&a.weight, p)?;
        let target = match a.family.as_str() {
            "zero" => &mut zero,
            "lattice" => &mut lattice,
            _ => bail!("unknown weighted atom family"),
        };
        let values = basis
            .iter()
            .map(|b| {
                b.iter()
                    .rev()
                    .fold(Float::with_val(p, 0), |v, c| v * &x + c)
            })
            .collect::<Vec<_>>();
        for row in 0..n {
            for col in 0..=row {
                let v = Float::with_val(p, &values[row]) * &values[col] * &weight;
                target[row * n + col] += &v;
                if row != col {
                    target[col * n + row] += v;
                }
            }
        }
    }
    let tail = if let Some(v) = tail_correction {
        if v.len() != n * n {
            bail!("tail correction shape mismatch");
        }
        coeffs(v, p)?
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
