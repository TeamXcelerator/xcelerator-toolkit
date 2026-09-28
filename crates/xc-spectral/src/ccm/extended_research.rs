//! Additional source-bound point diagnostics. External reference values are data,
//! never executable target definitions. No new primary eigensolve is available.
use super::retained_evidence::{
    managed, matrix_ancestry, point, precision, scalar, solve_gram_checked, transform_terms,
    ResearchRecord, RetainedMatrix, RetainedRoots,
};
use super::state_geometry::RetainedState;
use anyhow::{bail, Result};
use rayon::prelude::*;
use rug::{float::Constant, Float};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};
use xc_cache::*;
use xc_numerics::prefix::lossless_decimal as dec;

#[path = "extended_research/allowance_math.rs"]
mod allowance_math;
#[path = "extended_research/atom_math.rs"]
pub(super) mod atom_math;
#[path = "extended_research/cluster_math.rs"]
mod cluster_math;
#[path = "extended_research/compactness_math.rs"]
mod compactness_math;
#[path = "extended_research/directional_math.rs"]
mod directional_math;
#[path = "extended_research/energy_math.rs"]
mod energy_math;
#[path = "extended_research/weighted_math.rs"]
mod weighted_math;

pub const INPUT_KIND: &str = "ccm_external_research_source";
pub const DIAGNOSTICS: &[&str] = &[
    "compactness",
    "weighted_reference_projection",
    "signed_transform",
    "arithmetic_energy",
    "directional_response",
    "weighted_tail",
    "spectral_cluster",
    "resolution_budget",
    "energy_allowance",
];
pub fn artifact_kind(id: &str) -> Option<&'static str> {
    Some(match id {
        "compactness" => "ccm_compactness_analysis",
        "weighted_reference_projection" => "ccm_weighted_reference_projection",
        "signed_transform" => "ccm_signed_transform_analysis",
        "arithmetic_energy" => "ccm_arithmetic_energy_analysis",
        "directional_response" => "ccm_directional_response_analysis",
        "weighted_tail" => "ccm_weighted_tail_analysis",
        "spectral_cluster" => "ccm_spectral_cluster_analysis",
        "resolution_budget" => "ccm_resolution_budget_analysis",
        "energy_allowance" => "ccm_energy_allowance_analysis",
        "complex_transform" => "ccm_complex_transform_analysis",
        "root_transport" => "ccm_root_transport_analysis",
        "operator_cluster" => "ccm_operator_cluster_analysis",
        "finite_section_transfer" => "ccm_finite_section_transfer",
        "tail_operator" => "ccm_tail_operator_analysis",
        "observable_budget" => "ccm_observable_budget_analysis",
        "capture_preflight" => "ccm_capture_preflight",
        "consistency" => "ccm_consistency_analysis",
        "configuration_comparison" => "ccm_configuration_comparison",
        "band_reconstruction" => "ccm_band_reconstruction",
        "transform_enclosure" => "ccm_transform_enclosure",
        _ => return None,
    })
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ExtensionOptions {
    pub working_precision_bits: u32,
    pub maximum_rows: usize,
    pub maximum_directional_rows: usize,
    pub maximum_estimated_output_bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub maximum_working_bytes: Option<u64>,
    pub exponential_rates: Vec<String>,
    pub relative_tolerance: String,
}
impl ExtensionOptions {
    pub fn for_source(s: &RetainedState) -> Self {
        Self {
            working_precision_bits: s.precision.saturating_add(64).min(1_000_000),
            maximum_rows: 100_000,
            maximum_directional_rows: 16,
            maximum_estimated_output_bytes: 256 * 1024 * 1024,
            maximum_working_bytes: None,
            exponential_rates: vec!["0.5".into(), "1".into()],
            relative_tolerance: "0.01".into(),
        }
    }
    fn validate(&self, s: &RetainedState) -> Result<()> {
        precision(self.working_precision_bits)?;
        if self.working_precision_bits < s.precision
            || self.maximum_rows == 0
            || self.maximum_directional_rows > self.maximum_rows
            || self.maximum_rows > 100000
            || self.exponential_rates.len() > 8
        {
            bail!("invalid extended research precision or resource budget");
        }
        for a in &self.exponential_rates {
            let a = scalar(a, self.working_precision_bits)?;
            if !(0..=100).contains(&a) {
                bail!("exponential rate outside supported range");
            }
        }
        if self.maximum_estimated_output_bytes == 0 || self.maximum_working_bytes == Some(0) {
            bail!("output budget must be positive");
        }
        let t = scalar(&self.relative_tolerance, self.working_precision_bits)?;
        if t <= 0 || t >= 1 {
            bail!("resolution tolerance must lie between zero and one");
        }
        Ok(())
    }
}
/// Caller-supplied numerical reference. The particular target formula is never
/// required or copied. Nodes are x=j*log(C)/(2*intervals), j=0..intervals.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SampledReference {
    pub definition_digest: ContentDigest,
    pub evaluation_policy: String,
    pub approximation_scope: String,
    pub intervals: usize,
    pub values: Vec<String>,
    pub basis_values: Vec<Vec<String>>,
    pub fixed_second_component: Option<String>,
    pub raw_normalizer: String,
    pub trial_coefficients: Option<Vec<String>>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Jet {
    pub value: String,
    pub derivative: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReferenceJet {
    /// Explicit caller-declared join to the authenticated retained root source.
    /// Omission forbids treating a window ordinal as a reference ordinal.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub matched_root_ordinal: Option<usize>,
    pub ordinal: usize,
    pub t: String,
    pub reference_window: Jet,
    pub reference_full: Jet,
    pub exterior_tail: Jet,
    pub endpoint_tail_part: Option<Jet>,
    pub fitted_interior_parts: Vec<Jet>,
    /// Absolute point-input allowances, not certified by this producer.
    #[serde(default)]
    pub error_normalization: Option<String>,
    pub source_value_error: Option<String>,
    pub tail_value_error: Option<String>,
    pub source_derivative_error: Option<String>,
    pub root_separation_radius: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RankOneTerm {
    pub weight: String,
    pub vector: Vec<String>,
}
/// Symmetric matrix in the exact full -N..N coefficient coordinates. A sum of
/// diagonal/dense/rank-one parts avoids forcing dense storage for simple terms.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct OperatorComponent {
    pub label: String,
    pub source_digest: ContentDigest,
    pub diagonal: Vec<String>,
    pub dense: Vec<String>,
    pub rank_one: Vec<RankOneTerm>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct WeightedAtom {
    pub ordinal: usize,
    pub coordinate: String,
    pub weight: String,
    pub family: String,
    pub partition: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ClusterVector {
    pub source_digest: ContentDigest,
    pub n_modes: usize,
    pub precision_bits: u32,
    pub eigenvalue: String,
    pub coefficients: Vec<String>,
    pub assembly_policy: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EnergyAllowance {
    pub upper_trial_energy: String,
    pub low_block_lower_bound: String,
    pub high_block_lower_bound: String,
    pub cross_block_norm_bound: String,
    pub hypothesis_record_digest: ContentDigest,
    pub hypotheses: Vec<String>,
}
/// Explicit non-executable numerical inputs. All quantities refer to the named
/// state; this bundle does not authenticate externally asserted error bounds.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ExternalResearchInputs {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_once: Option<super::convergence_capture::RunOnceInputs>,
    pub schema_version: u32,
    pub source_eigenpair: ContentDigest,
    pub lambda_squared: String,
    pub n_modes: usize,
    pub precision_bits: u32,
    pub convention_id: String,
    pub definition_digest: ContentDigest,
    pub approximation_scope: String,
    #[serde(default)]
    pub target: Option<SampledReference>,
    #[serde(default)]
    pub reference_jets: Vec<ReferenceJet>,
    /// Explicit arithmetic-energy operators. When nonempty, these take
    /// precedence over `run_once.component_actions`; the two representations
    /// are alternatives and are not added together.
    #[serde(default)]
    pub components: Vec<OperatorComponent>,
    #[serde(default)]
    pub components_are_complete: bool,
    #[serde(default)]
    pub perturbations: Vec<OperatorComponent>,
    #[serde(default)]
    pub deficit: Option<String>,
    #[serde(default)]
    pub deficit_kind: Option<String>,
    #[serde(default)]
    pub atoms: Vec<WeightedAtom>,
    #[serde(default)]
    pub atom_coordinate: Option<String>,
    #[serde(default)]
    pub atom_coverage: Option<String>,
    #[serde(default)]
    pub tail_checkpoints: Vec<String>,
    #[serde(default)]
    pub cluster: Vec<ClusterVector>,
    #[serde(default)]
    pub previous_cluster: Vec<ClusterVector>,
    #[serde(default)]
    pub cluster_boundary_eigenvalues: Option<[String; 2]>,
    #[serde(default)]
    pub energy_allowance: Option<EnergyAllowance>,
}
impl ExternalResearchInputs {
    fn arithmetic_component_count(&self) -> usize {
        if self.components.is_empty() {
            self.run_once
                .as_ref()
                .map_or(0, |inputs| inputs.component_actions.len())
        } else {
            self.components.len()
        }
    }

    pub fn validate(&self) -> Result<()> {
        precision(self.precision_bits)?;
        if self.schema_version != 1
            || self.n_modes > 8192
            || scalar(&self.lambda_squared, self.precision_bits)? <= 1
            || !self.source_eigenpair.validate()
            || !self.definition_digest.validate()
            || self.convention_id.trim().is_empty()
            || self.convention_id.len() > 256
            || self.approximation_scope.trim().is_empty()
            || self.approximation_scope.len() > 16384
            || self.reference_jets.len() > 100000
            || self.atoms.len()
                > super::atom_research::policy(Some(self)).map_or(100000, |a| a.maximum_atoms)
            || self.components.len() > 64
            || self.perturbations.len() > 64
            || self.cluster.len() > 32
            || self.previous_cluster.len() > 32
            || self.tail_checkpoints.len() > 1024
        {
            bail!("invalid external research input identity or budget");
        }
        let p = self.precision_bits;
        let n = 2 * self.n_modes + 1;
        if let Some(inputs) = &self.run_once {
            inputs.validate(n, p)?;
        }
        let mut count = 0usize;
        let mut check = |v: &str| -> Result<()> {
            scalar(v, p)?;
            count += 1;
            Ok(())
        };
        if let Some(t) = &self.target {
            if !t.definition_digest.validate()
                || t.evaluation_policy.is_empty()
                || t.approximation_scope.is_empty()
                || t.intervals < 8
                || t.intervals > 131072
                || t.values.len() != t.intervals + 1
                || t.basis_values.len() > 8
                || t.basis_values.iter().any(|v| v.len() != t.values.len())
                || t.fixed_second_component.is_some() && t.basis_values.len() != 2
                || t.trial_coefficients.as_ref().is_some_and(|v| v.len() != n)
            {
                bail!("invalid sampled reference shape or definition");
            }
            for v in t
                .values
                .iter()
                .chain(t.basis_values.iter().flatten())
                .chain(t.trial_coefficients.iter().flatten())
                .chain(t.fixed_second_component.iter())
                .chain(std::iter::once(&t.raw_normalizer))
            {
                check(v)?;
            }
            if scalar(&t.raw_normalizer, p)? == 0 {
                bail!("reference raw normalizer is zero");
            }
        }
        let mut seen = BTreeSet::new();
        for r in &self.reference_jets {
            if r.ordinal == 0
                || r.matched_root_ordinal == Some(0)
                || !seen.insert(r.ordinal)
                || r.fitted_interior_parts.len() > 8
            {
                bail!("invalid reference ordinal");
            }
            check(&r.t)?;
            if (r.source_value_error.is_some()
                || r.source_derivative_error.is_some()
                || r.root_separation_radius.is_some())
                && r.error_normalization.as_deref() != Some("unit_l2_dx")
            {
                bail!("root-budget error allowances require explicit unit_l2_dx normalization");
            }

            for j in [&r.reference_window, &r.reference_full, &r.exterior_tail]
                .into_iter()
                .chain(r.endpoint_tail_part.iter())
                .chain(r.fitted_interior_parts.iter())
            {
                check(&j.value)?;
                check(&j.derivative)?;
            }
            for v in r
                .source_value_error
                .iter()
                .chain(r.tail_value_error.iter())
                .chain(r.source_derivative_error.iter())
                .chain(r.root_separation_radius.iter())
            {
                check(v)?;
                if scalar(v, p)? < 0 {
                    bail!("negative declared error allowance");
                }
            }
        }
        for set in [&self.components, &self.perturbations] {
            let mut labels = BTreeSet::new();
            for c in set {
                if c.label.is_empty()
                    || c.label.len() > 256
                    || !labels.insert(&c.label)
                    || !c.source_digest.validate()
                    || !c.diagonal.is_empty() && c.diagonal.len() != n
                    || !c.dense.is_empty() && c.dense.len() != n * n
                    || c.rank_one.len() > 128
                    || c.rank_one.iter().any(|t| t.vector.len() != n)
                {
                    bail!("invalid operator component");
                }
                for v in c.diagonal.iter().chain(c.dense.iter()).chain(
                    c.rank_one
                        .iter()
                        .flat_map(|t| t.vector.iter().chain(std::iter::once(&t.weight))),
                ) {
                    check(v)?;
                }
                if !c.dense.is_empty() {
                    for i in 0..n {
                        for j in 0..i {
                            if scalar(&c.dense[i * n + j], p)? != scalar(&c.dense[j * n + i], p)? {
                                bail!("component matrix is not symmetric");
                            }
                        }
                    }
                }
            }
        }
        let mut atoms = BTreeSet::new();
        for a in &self.atoms {
            if a.ordinal == 0
                || !["zero", "lattice"].contains(&a.family.as_str())
                || a.partition.is_empty()
                || a.partition.len() > 128
                || !atoms.insert((&a.family, a.ordinal))
                || scalar(&a.coordinate, p)? < 0
                || (a.family == "zero" && scalar(&a.coordinate, p)? == 0)
            {
                bail!("invalid weighted atom");
            }
            check(&a.coordinate)?;
            check(&a.weight)?;
        }
        if !self.atoms.is_empty()
            && (self.atom_coordinate.as_ref().is_none_or(|v| v.is_empty())
                || self.atom_coverage.as_ref().is_none_or(|v| v.is_empty()))
        {
            bail!("atom coordinate and coverage declarations are required");
        }
        for c in self.cluster.iter().chain(&self.previous_cluster) {
            precision(c.precision_bits)?;
            if c.precision_bits > self.precision_bits {
                bail!("external precision must cover all cluster vectors");
            }
            if c.n_modes > 8192
                || c.coefficients.len() != 2 * c.n_modes + 1
                || !c.source_digest.validate()
                || c.assembly_policy.is_empty()
            {
                bail!("invalid cluster vector");
            }
            for v in c.coefficients.iter().chain(std::iter::once(&c.eigenvalue)) {
                check(v)?;
            }
            if c.coefficients
                .iter()
                .all(|v| scalar(v, p).is_ok_and(|x| x == 0))
            {
                bail!("zero cluster vector");
            }
        }
        for v in self
            .tail_checkpoints
            .iter()
            .chain(self.deficit.iter())
            .chain(self.cluster_boundary_eigenvalues.iter().flatten())
        {
            check(v)?;
        }
        if self.deficit.is_some()
            && !matches!(
                self.deficit_kind.as_deref(),
                Some("exact_source" | "fuchs_approximation" | "other_approximation")
            )
        {
            bail!("deficit source kind must be explicit");
        }
        if let Some(a) = &self.energy_allowance {
            if !a.hypothesis_record_digest.validate() || a.hypotheses.is_empty() {
                bail!("allowance hypothesis record is required");
            }
            for v in [
                &a.upper_trial_energy,
                &a.low_block_lower_bound,
                &a.high_block_lower_bound,
                &a.cross_block_norm_bound,
            ] {
                check(v)?;
            }
            if scalar(&a.cross_block_norm_bound, p)? < 0 {
                bail!("negative operator norm bound");
            }
        }
        if count
            > super::atom_research::policy(Some(self)).map_or(4_000_000, |a| {
                a.maximum_atoms.saturating_mul(2).saturating_add(4_000_000)
            })
        {
            bail!("external scalar budget exceeded");
        }
        xc_core::validate_secret_free(self, "external research values")?;
        Ok(())
    }
    pub(crate) fn matches(&self, s: &RetainedState) -> Result<()> {
        self.validate()?;
        if self.source_eigenpair != s.manifest.content_digest
            || self.lambda_squared != s.cutoff
            || self.n_modes != s.modes
        {
            bail!("external research input belongs to another retained state");
        }
        Ok(())
    }
    pub fn from_file(path: &std::path::Path) -> Result<Self> {
        if std::fs::metadata(path)?.len() > 64 * 1024 * 1024 {
            bail!("external research input exceeds 64 MiB");
        }
        Self::from_bytes(path, &std::fs::read(path)?)
    }
    /// Decode exactly these bytes; path only resolves separately hashed atom chunks.
    pub fn from_bytes(path: &std::path::Path, bytes: &[u8]) -> Result<Self> {
        if bytes.len() > 64 * 1024 * 1024 {
            bail!("external research input exceeds 64 MiB");
        }
        let mut v: Self = serde_json::from_slice(bytes)?;
        super::atom_research::expand_tables(
            path,
            &mut v.atoms,
            v.run_once.as_mut().and_then(|r| r.completion.as_mut()),
            v.precision_bits,
        )?;
        v.validate()?;
        Ok(v)
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisRow {
    pub ordinal: usize,
    pub label: String,
    pub outcome: String,
    pub values: BTreeMap<String, String>,
    pub notes: Vec<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExtendedAnalysis {
    pub diagnostic: String,
    pub outcome: String,
    pub reason: Option<String>,
    pub lambda_squared: String,
    pub n_modes: usize,
    pub source_precision_bits: u32,
    pub working_precision_bits: u32,
    pub convention: String,
    pub assurance: String,
    pub values: BTreeMap<String, String>,
    pub rows: Vec<AnalysisRow>,
}
const WEIGHTED_PROFILE_ASSURANCE: &str = "finite_grid_arithmetic_enclosures; exact stored source/reference points; excludes source-construction and quadrature errors; no continuum or limit claim";
const DIRECTIONAL_ASSURANCE: &str = "finite_stored_point_arithmetic_enclosures; response ratios conditional on root, simple-minimum and displacement hypotheses; no source-error or ground-selection certificate";
const TRANSPORT_ASSURANCE: &str = "finite_stored_point_arithmetic_enclosures; inherited directional intervals; conditional root/displacement hypotheses; no source-error or root certificate";
const COMPLEX_ASSURANCE: &str = "finite_stored_point_arithmetic_enclosures; exact cutoff and retained ordinates; samples do not certify contour counts, roots or source errors";
const OPERATOR_CLUSTER_ASSURANCE: &str = "finite_stored_point_arithmetic_enclosures; numerical selected subspace only; no full-declared-span, source-error or spectral-selection certificate";
const TRANSFER_ASSURANCE: &str = "finite_stored_point_arithmetic_enclosures; projected retained state only; external comparison source not certified; no convergence claim";
const CONSISTENCY_ASSURANCE: &str = "finite_stored_point_arithmetic_enclosures; common signed unit-state convention and source independence remain external premises";
const OBSERVATION_ASSURANCE: &str = "finite_stored_point_arithmetic_enclosures; conditional finite-support L2 transport; external state-error premise not certified";
const COMPARISON_ASSURANCE: &str = "finite_stored_point_arithmetic_enclosures; external comparison source and selection premises not certified; no convergence claim";
const RESOLUTION_ASSURANCE: &str = "finite_stored_point_arithmetic_enclosures; conditional root distance bounds; external source errors and target isolation or curvature hypotheses not certified";
const ALLOWANCE_ASSURANCE: &str = "finite_stored_point_arithmetic_enclosures; conditional block expressions; external block bounds and hypotheses not certified";
const ENERGY_ASSURANCE: &str = "finite_stored_point_arithmetic_enclosures; excludes source-construction and operator-model errors; no ground-selection or convergence claim";
const ASSURANCE: &str = "point_diagnostics_only; external_inputs_and_allowances_not_certified; no_RH_or_ground_selection_premise";
pub(super) fn report(id: &str, s: &RetainedState, o: &ExtensionOptions) -> ExtendedAnalysis {
    ExtendedAnalysis {
        diagnostic: id.into(),
        outcome: "point_measurement".into(),
        reason: None,
        lambda_squared: s.cutoff.clone(),
        n_modes: s.modes,
        source_precision_bits: s.precision,
        working_precision_bits: o.working_precision_bits,
        convention:
            "centered_full_V_fourier; exact_source_identity; explicit_reference_normalization"
                .into(),
        assurance: if id == "transform_enclosure" {
            "finite_retained_function_enclosures; source_scope_explicit; no_infinite_limit_claim"
        } else if id == "directional_response" {
            DIRECTIONAL_ASSURANCE
        } else if id == "root_transport" {
            TRANSPORT_ASSURANCE
        } else if id == "complex_transform" {
            COMPLEX_ASSURANCE
        } else if id == "operator_cluster" {
            OPERATOR_CLUSTER_ASSURANCE
        } else if id == "finite_section_transfer" {
            TRANSFER_ASSURANCE
        } else if id == "consistency" {
            CONSISTENCY_ASSURANCE
        } else if id == "observable_budget" {
            OBSERVATION_ASSURANCE
        } else if id == "configuration_comparison" {
            COMPARISON_ASSURANCE
        } else if id == "resolution_budget" {
            RESOLUTION_ASSURANCE
        } else if id == "energy_allowance" {
            ALLOWANCE_ASSURANCE
        } else if id == "arithmetic_energy" {
            ENERGY_ASSURANCE
        } else if id == "weighted_reference_projection" {
            WEIGHTED_PROFILE_ASSURANCE
        } else {
            ASSURANCE
        }
        .into(),
        values: BTreeMap::new(),
        rows: vec![],
    }
}
pub(super) fn row(ordinal: usize, label: impl Into<String>) -> AnalysisRow {
    AnalysisRow {
        ordinal,
        label: label.into(),
        outcome: "point_measurement".into(),
        values: BTreeMap::new(),
        notes: vec![],
    }
}
pub(super) fn put(m: &mut BTreeMap<String, String>, name: &str, v: &Float) {
    m.insert(name.into(), dec(v));
}
pub(super) fn missing(mut r: ExtendedAnalysis, reason: &str) -> ExtendedAnalysis {
    r.outcome = "missing_input".into();
    r.reason = Some(reason.into());
    r
}
pub(super) fn unresolved(mut r: ExtendedAnalysis, reason: &str) -> ExtendedAnalysis {
    r.outcome = "unresolved".into();
    r.reason = Some(reason.into());
    r
}
pub(super) fn coeffs(v: &[String], p: u32) -> Result<Vec<Float>> {
    v.iter().map(|x| scalar(x, p)).collect()
}
pub(super) fn source_unit(s: &RetainedState, p: u32) -> Result<Vec<Float>> {
    compactness_math::unit(&s.coefficients, p)
}
pub(super) fn evaluate(v: &[Float], x: &Float, l: &Float, p: u32) -> (Float, Float) {
    let n = v.len() / 2;
    let angle = Float::with_val(p, Constant::Pi) * (Float::with_val(p, x) * 2u32 / l + 1u32);
    let (sn, cs) = angle.sin_cos(Float::new(p));
    let mut re = Float::with_val(p, &v[n]);
    let mut im = Float::with_val(p, 0);
    let mut sin = Float::with_val(p, 0);
    let mut cos = Float::with_val(p, 1);
    for j in 1..=n {
        let next_s = Float::with_val(p, &sin) * &cs + Float::with_val(p, &cos) * &sn;
        cos = Float::with_val(p, &cos) * &cs - Float::with_val(p, &sin) * &sn;
        sin = next_s;
        re += (Float::with_val(p, &v[n + j]) + &v[n - j]) * &cos;
        im += (Float::with_val(p, &v[n + j]) - &v[n - j]) * &sin;
    }
    (re, im)
}
fn compactness(s: &RetainedState, o: &ExtensionOptions) -> Result<ExtendedAnalysis> {
    let mut r = report("compactness", s, o);
    let p = o.working_precision_bits;
    // Conservative additional scratch estimate includes the maximum retry guard.
    // It does not purport to bound the host process or an independently held matrix.
    let scratch = 32u64
        * (s.coefficients.len() as u64 + 64)
        * ((u64::from(p.max(s.precision)) + 4096).div_ceil(8) + 128);
    if o.maximum_working_bytes.is_some_and(|limit| scratch > limit) {
        return Ok(unresolved(
            r,
            "compactness arithmetic scratch estimate exceeds explicit working-byte budget",
        ));
    }
    let measured =
        match compactness_math::measure(&s.coefficients, &s.cutoff, &o.exponential_rates, p) {
            Ok(measured) => measured,
            Err(error) => {
                return Ok(unresolved(
                    r,
                    &format!("finite moment arithmetic unresolved: {error}; no zero inferred"),
                ))
            }
        };
    put(&mut r.values, "transform_origin", &measured.origin);
    put(
        &mut r.values,
        "transform_second_derivative",
        &measured.second,
    );
    put(
        &mut r.values,
        "transform_fourth_derivative",
        &measured.fourth,
    );
    put(
        &mut r.values,
        "arithmetic_precision_bits",
        &Float::with_val(p, measured.arithmetic_precision),
    );
    if let Some(sigma) = measured.sigma {
        put(&mut r.values, "sigma", &sigma);
    } else {
        r.outcome = "partial_unresolved".into();
        r.reason = Some("origin transform is exactly zero; sigma is undefined".into());
    }
    for (i, value) in measured.weighted.into_iter().enumerate() {
        let mut rr = row(i + 1, "finite_exponential_weighted_norm_squared");
        put(&mut rr.values, "rate", &value.rate);
        put(&mut rr.values, "analytic_integral", &value.value);
        put(&mut rr.values, "sum_absolute_terms", &value.absolute_terms);
        rr.notes.push("point accepted when directed finite-state arithmetic enclosures agree after rounding; source construction error and infinite-support behavior are not enclosed".into());
        r.rows.push(rr);
    }
    r.convention = "unit_L2_dx; finite Fourier origin moments and exp(2*a*abs(x)) norm; directed arithmetic enclosures agree at requested rounding; source construction errors excluded".into();
    Ok(r)
}
fn weighted_projection(
    s: &RetainedState,
    o: &ExtensionOptions,
    i: Option<&ExternalResearchInputs>,
) -> Result<ExtendedAnalysis> {
    use rug::float::Round;
    let mut r = report("weighted_reference_projection", s, o);
    let Some((input, t)) = i.and_then(|i| i.target.as_ref().map(|t| (i, t))) else {
        return Ok(missing(
            r,
            "external sampled target and projection convention required",
        ));
    };
    if s.coefficients
        .iter()
        .zip(s.coefficients.iter().rev())
        .any(|(a, b)| a != b)
    {
        return Ok(unresolved(
            r,
            "weighted real-even profile convention requires exactly even coefficients",
        ));
    }
    let p = o.working_precision_bits;
    let base = p.max(s.precision).max(input.precision_bits);
    let scratch = 64u64
        * (s.coefficients.len() as u64
            + (t.basis_values.len() as u64 + 6) * (t.values.len() as u64)
            + 64)
        * ((u64::from(base) + 4096).div_ceil(8) + 128);
    if o.maximum_working_bytes.is_some_and(|limit| scratch > limit) {
        return Ok(unresolved(
            r,
            "weighted profile scratch estimate exceeds explicit working-byte budget",
        ));
    }
    // These decimals denote stored source points, not exact real parameters.
    let target = coeffs(&t.values, input.precision_bits)?;
    let basis = t
        .basis_values
        .iter()
        .map(|v| coeffs(v, input.precision_bits))
        .collect::<Result<Vec<_>>>()?;
    let fixed = t
        .fixed_second_component
        .as_ref()
        .map(|v| scalar(v, input.precision_bits))
        .transpose()?;
    if target[0] != 1 {
        return Ok(unresolved(
            r,
            "reference source point must be normalized to target(1)=1 exactly",
        ));
    }
    let measured = match weighted_math::measure(
        &s.coefficients,
        &s.cutoff,
        &target,
        &basis,
        fixed.as_ref(),
        p,
    ) {
        Ok(measured) => measured,
        Err(error) => {
            return Ok(unresolved(
                r,
                &format!("weighted profile unresolved: {error}"),
            ))
        }
    };
    for (name, bound) in measured.values {
        bound.validate()?;
        let middle = bound.midpoint_point();
        let mid = Float::with_val(p, middle.lower());
        let lo = Float::with_val_round(p, bound.lower(), Round::Down).0;
        let hi = Float::with_val_round(p, bound.upper(), Round::Up).0;
        if !mid.is_finite()
            || !lo.is_finite()
            || !hi.is_finite()
            || (mid.is_zero() && !middle.lower().is_zero())
        {
            bail!("weighted profile report value exceeds requested output range");
        }
        put(&mut r.values, &name, &mid);
        // Directed decimal strings retain outwardness as real decimals too.
        let digits = Some((u64::from(p) * 30103 / 100000 + 10) as usize);
        r.values.insert(
            format!("{name}_lower"),
            lo.to_string_radix_round(10, digits, Round::Down),
        );
        r.values.insert(
            format!("{name}_upper"),
            hi.to_string_radix_round(10, digits, Round::Up),
        );
    }
    put(
        &mut r.values,
        "reference_raw_normalizer",
        &Float::with_val(p, scalar(&t.raw_normalizer, input.precision_bits)?),
    );
    if let Some(fixed) = fixed {
        put(
            &mut r.values,
            "fixed_second_component",
            &Float::with_val(p, fixed),
        );
    }
    put(
        &mut r.values,
        "arithmetic_precision_bits",
        &Float::with_val(p, measured.arithmetic_precision),
    );
    r.convention="f(1)=target(1)=1; x=j*log(C)/(2n); exp(x/2)dx; finite composite trapezoid; independently scaled raw basis; each measured field is an enclosure midpoint with outward decimal lower/upper fields; pivot uses scaled Gram units".into();
    r.assurance = WEIGHTED_PROFILE_ASSURANCE.into();
    if !measured.fit_resolved {
        r.outcome = "rank_or_precision_unresolved".into();
        r.reason = Some(
            "finite interval Gram solve unresolved within 4096 guard bits; fit fields withheld"
                .into(),
        );
    } else if t.fixed_second_component.is_some() && !r.values.contains_key("b_effective") {
        r.reason = Some("b_effective withheld because a_0 enclosure contains zero".into());
    }
    Ok(r)
}

// Range-checked point division; this is not a forward-error enclosure.
fn signed_channel_quotient(a: &Float, b: &Float, p: u32) -> Result<Float> {
    point::quotient(a, b, p)
}

fn signed_transforms(
    s: &RetainedState,
    o: &ExtensionOptions,
    input: Option<&ExternalResearchInputs>,
) -> Result<ExtendedAnalysis> {
    let mut r = report("signed_transform", s, o);
    let Some(i) = input.filter(|i| !i.reference_jets.is_empty()) else {
        return Ok(missing(
            r,
            "external reference window/full/tail jets required",
        ));
    };
    // Decimal input text denotes a binary point at the declared input precision.
    // Promote that existing point; changing output precision cannot redefine it.
    let p = o.working_precision_bits;
    let input_point =
        |text: &str| -> Result<Float> { point::output(&scalar(text, i.precision_bits)?, p) };
    let normalization_precision = p + 64;
    let l = super::retained_evidence::finite_math::rounded_log_cutoff(
        &s.cutoff,
        normalization_precision,
    )?;
    let c = point::unit_center(&s.coefficients, normalization_precision)?;
    if c.is_zero() {
        return Ok(unresolved(
            r,
            "center-one normalization requires a nonzero exact stored center",
        ));
    }
    let c = signed_channel_quotient(&c, &l.sqrt(), normalization_precision)?;
    r.convention="center_one; exp(i*t*x); interior=actual-reference_window; signed_total=interior-exterior_tail; reference_full=window+tail; all channels kept".into();
    put(
        &mut r.values,
        "arithmetic_precision_bits",
        &Float::with_val(p, p),
    );
    put(
        &mut r.values,
        "normalization_precision_bits",
        &Float::with_val(p, normalization_precision),
    );
    r.rows = i
        .reference_jets
        .par_iter()
        .map(|a| -> Result<_> {
            let t = input_point(&a.t)?;
            let (v, d, abs, absd) = transform_terms(s, &t, p)?;
            let mut rr = row(a.ordinal, "reference_ordinate");
            put(&mut rr.values, "t", &t);
            for (name, value, absolute, w, f, tail, end, fit) in [
                (
                    "value",
                    v,
                    abs,
                    &a.reference_window.value,
                    &a.reference_full.value,
                    &a.exterior_tail.value,
                    a.endpoint_tail_part.as_ref().map(|x| &x.value),
                    a.fitted_interior_parts
                        .iter()
                        .map(|x| &x.value)
                        .collect::<Vec<_>>(),
                ),
                (
                    "derivative",
                    d,
                    absd,
                    &a.reference_window.derivative,
                    &a.reference_full.derivative,
                    &a.exterior_tail.derivative,
                    a.endpoint_tail_part.as_ref().map(|x| &x.derivative),
                    a.fitted_interior_parts
                        .iter()
                        .map(|x| &x.derivative)
                        .collect::<Vec<_>>(),
                ),
            ] {
                let actual = signed_channel_quotient(&value, &c, p)?;
                let window = input_point(w)?;
                let full = input_point(f)?;
                let tail = input_point(tail)?;
                let inside = point::sum(&[actual.clone(), -window.clone()], p)?;
                let total = point::sum(&[actual.clone(), -window.clone(), -tail.clone()], p)?;
                let closure = point::sum(&[full.clone(), -window.clone(), -tail.clone()], p)?;
                let mut unfitted_terms = vec![actual.clone(), -window.clone()];
                for (k, a) in fit.iter().enumerate() {
                    let v = input_point(a)?;
                    put(&mut rr.values, &format!("{name}_fitted_{k}"), &v);
                    unfitted_terms.push(-v);
                }
                for (label, val) in [
                    ("actual", &actual),
                    ("reference_window", &window),
                    ("reference_full", &full),
                    ("exterior_tail", &tail),
                    ("signed_interior", &inside),
                    ("signed_total", &total),
                    ("reference_closure_defect", &closure),
                ] {
                    put(&mut rr.values, &format!("{name}_{label}"), val);
                }
                put(
                    &mut rr.values,
                    &format!("{name}_unfitted_interior"),
                    &point::sum(&unfitted_terms, p)?,
                );
                let scale = point::sum(&[inside.clone().abs(), tail.clone().abs()], p)?;
                put(
                    &mut rr.values,
                    &format!("{name}_sum_absolute_channels"),
                    &scale,
                );
                if total != 0 && scale > 0 {
                    put(
                        &mut rr.values,
                        &format!("{name}_cancellation_digits"),
                        &point::sum(&[scale.clone().log10(), -total.clone().abs().log10()], p)?
                            .max(&Float::with_val(p, 0)),
                    );
                }
                if let Some(e) = end {
                    let e = input_point(e)?;
                    put(&mut rr.values, &format!("{name}_endpoint_tail_part"), &e);
                    put(
                        &mut rr.values,
                        &format!("{name}_remaining_tail"),
                        &point::sum(&[tail, -e], p)?,
                    );
                }
                let numerical_floor = point::sum(
                    &[
                        signed_channel_quotient(&absolute, &c.clone().abs(), p)?,
                        Float::with_val(p, 1),
                    ],
                    p,
                )? >> (p - 32);
                if total.clone().abs() <= numerical_floor {
                    rr.outcome = "cancellation_limited".into();
                }
            }
            rr.notes.push(
                "declared-precision reference points promoted without reinterpretation; each signed channel uses one rounded sum; cancellation_limited is a point-arithmetic heuristic, not an error bound; no infinite-tail or source-error certification"
                    .into(),
            );
            Ok(rr)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(r)
}
pub(super) fn save_arithmetic_enclosure(
    map: &mut BTreeMap<String, String>,
    name: &str,
    value: &xc_numerics::mpfr_interval::MpfrInterval,
    p: u32,
) -> Result<()> {
    use rug::float::Round;
    value.validate()?;
    let middle = value.midpoint_point();
    let mid = point::output(middle.lower(), p)?;
    let lo = Float::with_val_round(p, value.lower(), Round::Down).0;
    let hi = Float::with_val_round(p, value.upper(), Round::Up).0;
    if !lo.is_finite() || !hi.is_finite() {
        bail!("energy enclosure output exceeds finite range")
    }
    put(map, name, &mid);
    let digits = Some((u64::from(p) * 30103 / 100000 + 10) as usize);
    map.insert(
        format!("{name}_lower"),
        lo.to_string_radix_round(10, digits, Round::Down),
    );
    map.insert(
        format!("{name}_upper"),
        hi.to_string_radix_round(10, digits, Round::Up),
    );
    Ok(())
}

fn arithmetic_energy(
    s: &RetainedState,
    m: Option<&RetainedMatrix<'_>>,
    o: &ExtensionOptions,
    input: Option<&ExternalResearchInputs>,
) -> Result<ExtendedAnalysis> {
    let mut r = report("arithmetic_energy", s, o);
    let Some(i) = input.filter(|i| {
        !i.components.is_empty()
            || i.run_once
                .as_ref()
                .is_some_and(|r| !r.component_actions.is_empty())
    }) else {
        return Ok(missing(
            r,
            "explicit arithmetic component operators required; total Tau does not identify the split",
        ));
    };
    let Some(m) = m else {
        return Ok(missing(r, "retained Tau required for component closure"));
    };
    let p = o.working_precision_bits;
    let count = m.entries.len() as u64
        + s.coefficients.len() as u64 * 256
        + i.components
            .iter()
            .map(|c| {
                (c.diagonal.len()
                    + c.dense.len()
                    + c.rank_one.iter().map(|r| r.vector.len() + 1).sum::<usize>())
                    as u64
            })
            .sum::<u64>();
    let scratch = count
        .checked_mul(64)
        .and_then(|n| n.checked_mul((u64::from(p) + 4096).div_ceil(8) + 128))
        .ok_or_else(|| anyhow::anyhow!("arithmetic energy scratch estimate overflow"))?;
    if o.maximum_working_bytes.is_some_and(|limit| scratch > limit) {
        return Ok(unresolved(
            r,
            "arithmetic energy scratch estimate exceeds explicit working-byte budget",
        ));
    }
    let Some(measured) = energy_math::measure(s, m, i, p)? else {
        return Ok(unresolved(
            r,
            "arithmetic energy enclosures unresolved within 4096 guard bits; measurements withheld",
        ));
    };
    for (name, value) in &measured.values {
        save_arithmetic_enclosure(&mut r.values, name, value, p)?;
    }
    put(
        &mut r.values,
        "arithmetic_precision_bits",
        &Float::with_val(p, measured.arithmetic_precision),
    );
    for (k, energy) in measured.energies.iter().enumerate() {
        let (label, digest, convention) = if let Some(c) = i.components.get(k) {
            (
                &c.label,
                &c.source_digest,
                "external operator on exact stored coefficients",
            )
        } else {
            let c = &i.run_once.as_ref().unwrap().component_actions[k];
            (&c.label, &c.source_digest, c.convention.as_str())
        };
        let mut rr = row(k + 1, label);
        save_arithmetic_enclosure(&mut rr.values, "energy", energy, p)?;
        rr.notes
            .push(format!("source {}; {}", digest.0, convention));
        r.rows.push(rr);
    }
    r.convention = if i.components_are_complete {
        "caller_declares_complete_component_sum; measured_action_closure"
    } else {
        "partial_component_sum; remaining_Tau_action_explicit"
    }
    .into();
    if i.deficit.is_some() {
        r.convention.push_str(&format!(
            "; reference_deficit_kind={}",
            i.deficit_kind.as_deref().unwrap_or("unspecified")
        ));
    }
    let mut reasons = Vec::new();
    if measured.zero_trial {
        reasons.push("zero trial vector has no Rayleigh quotient; trial fields omitted");
    }
    if i.deficit
        .as_ref()
        .is_some_and(|d| scalar(d, i.precision_bits).is_ok_and(|d| d <= 0))
    {
        reasons.push("nonpositive reference deficit; deficit ratios omitted");
    }
    if !reasons.is_empty() {
        r.outcome = "partial_unresolved".into();
        r.reason = Some(reasons.join("; "));
    }
    Ok(r)
}

fn directional(
    s: &RetainedState,
    m: Option<&RetainedMatrix<'_>>,
    roots: Option<&RetainedRoots>,
    o: &ExtensionOptions,
    input: Option<&ExternalResearchInputs>,
) -> Result<ExtendedAnalysis> {
    let mut r = report("directional_response", s, o);
    let Some(m) = m else {
        return Ok(missing(r, "retained Tau required"));
    };
    let Some(roots) = roots else {
        return Ok(missing(r, "retained root window required"));
    };
    if s.coefficients
        .iter()
        .zip(s.coefficients.iter().rev())
        .any(|(a, b)| a != b)
    {
        return Ok(unresolved(
            r,
            "directional convention requires exactly even coefficients",
        ));
    }
    let p = o.working_precision_bits;
    if let Some(limit) = o.maximum_working_bytes {
        let input_bytes = input
            .map(serde_json::to_vec)
            .transpose()?
            .map_or(0, |v| v.len());
        let cells = (m.entries.len() as u128
            + input_bytes as u128
            + 32 * s.coefficients.len() as u128
            + 256)
            * 16;
        let estimate = cells * (u128::from(p + 4096).div_ceil(8) + 64);
        if estimate > u128::from(limit) {
            return Ok(unresolved(
                r,
                "directional maximum-guard scratch estimate exceeds explicit working-byte budget",
            ));
        }
    }
    // Explicit arithmetic work budget. Every omitted row remains in the report.
    let limit = o.maximum_directional_rows;
    r.convention="even finite state; tau=(t*log(C)/(2*pi))^2; x=(tau-j^2)^-1*v-v*(v^T(tau-j^2)^-1*v); kappa=x^T(Tau-EI)x; response ratios conditional on simple-minimum/root/source hypotheses and measured ||(A-E)z-c*P*R*1||/||(A-E)z||<=2^(-source_precision/2), c least-squares; this numerical rule does not prove exact displacement".into();
    let checkpoints = super::capture_runtime::Checkpoints::new(&(
        "directional-rows-v4-displacement-rule",
        &s.manifest.content_digest,
        &m.manifest.content_digest,
        &roots.manifest.content_digest,
        o,
        input,
    ))?;
    r.rows = super::capture_runtime::row_blocks(
        &checkpoints,
        roots.dataset.points.len(),
        |idx| -> Result<_> {
            let point = &roots.dataset.points[idx];
            let mut rr = row(point.ordinal, "retained_window_ordinal");
            if idx >= limit {
                rr.outcome = "budget_limited".into();
                rr.notes.push(
                    "explicit maximum_directional_rows reached; no source recomputation".into(),
                );
                return Ok(rr);
            }
            let Some(t) = &point.value else {
                rr.outcome = "missing_input".into();
                return Ok(rr);
            };
            let measured = match directional_math::measure(
                s,
                m,
                input,
                t,
                roots.dataset.precision_bits,
                p,
            )? {
                directional_math::Outcome::Measured(value) => value,
                directional_math::Outcome::Carrier => {
                    rr.outcome = "carrier_or_unresolved".into();
                    rr.notes.push("zero retained point lies on the central carrier; projected resolvent undefined".into());
                    return Ok(rr);
                }
                directional_math::Outcome::GuardExhausted => {
                    rr.outcome = "cancellation_limited".into();
                    rr.notes.push("carrier separation or arithmetic width unresolved within 4096 guard bits; measurements withheld".into());
                    return Ok(rr);
                }
            };
            for (name, value) in &measured.values {
                save_arithmetic_enclosure(&mut rr.values, name, value, p)?;
            }
            put(
                &mut rr.values,
                "arithmetic_precision_bits",
                &Float::with_val(p, measured.precision),
            );
            if !measured.denominator_resolved {
                rr.outcome = "unresolved_denominator".into();
            }
            if !measured.root_rule_met || !measured.has_actions || !measured.displacement_rule_met {
                if rr.outcome == "point_measurement" {
                    rr.outcome = "channels_resolved_budget_unassessed".into();
                }
                rr.notes.push("root proximity or displacement-alignment rule not met, or perturbation actions unavailable; response ratios withheld".into());
            }
            rr.notes.push("finite stored-point arithmetic bounds only; the root proximity and displacement-alignment thresholds are numerical rules, not a root, minimum, exact displacement or spectral-gap certificate".into());
            rr.notes.push(format!(
                "source root status {}; point energy is not a spectral-gap certificate",
                point.source_status
            ));
            Ok(rr)
        },
    )?;
    Ok(r)
}
pub(crate) fn weighted_tail_base(
    s: &RetainedState,
    o: &ExtensionOptions,
    input: Option<&ExternalResearchInputs>,
) -> Result<ExtendedAnalysis> {
    let mut r = report("weighted_tail", s, o);
    let Some(i) = input.filter(|i| !i.atoms.is_empty()) else {
        return Ok(missing(
            r,
            "explicit weighted atoms, coordinate and coverage required",
        ));
    };
    let p = o.working_precision_bits;
    r.convention = format!(
        "{}; caller-declared partitions; finite supplied atoms only; {}; numeric atoms/checkpoints decoded at declared source precision; exact dyadic masses and directed inverse-moment arithmetic",
        i.atom_coordinate.as_deref().unwrap(),
        i.atom_coverage.as_deref().unwrap()
    );
    let mut checkpoints = i
        .tail_checkpoints
        .iter()
        .map(|x| scalar(x, i.precision_bits))
        .collect::<Result<Vec<_>>>()?;
    if checkpoints.is_empty() {
        for a in &i.atoms {
            checkpoints.push(scalar(&a.coordinate, i.precision_bits)?);
        }
        checkpoints.sort_by(Float::total_cmp);
        checkpoints.dedup();
        let count = checkpoints.len();
        checkpoints = checkpoints
            .into_iter()
            .enumerate()
            .filter(|(k, _)| (*k + 1).is_power_of_two() || *k + 1 == count)
            .map(|(_, v)| v)
            .collect();
    }
    checkpoints.sort_by(Float::total_cmp);
    checkpoints.dedup();
    if checkpoints.iter().any(|x| x < &0) {
        bail!("tail checkpoints must be nonnegative");
    }
    let mut families: BTreeMap<(String, String), Vec<atom_math::Atom>> = BTreeMap::new();
    for a in &i.atoms {
        families
            .entry((a.family.clone(), a.partition.clone()))
            .or_default()
            .push(atom_math::Atom {
                coordinate: scalar(&a.coordinate, i.precision_bits)?,
                weight: scalar(&a.weight, i.precision_bits)?,
            });
    }
    for ((family, partition), atoms) in families {
        if let Some(limit) = o.maximum_working_bytes {
            if atom_math::scratch_bytes(&atoms, checkpoints.len(), p)? > limit {
                return Ok(unresolved(
                    r,
                    "finite atom per-group scratch estimate exceeds explicit working-byte budget",
                ));
            }
        }
        let rows = atom_math::tail(&atoms, &checkpoints, p)?;
        for value in rows {
            let mut rr = row(r.rows.len() + 1, format!("{family}/{partition}"));
            put(&mut rr.values, "cutoff", &value.cutoff);
            put(&mut rr.values, "included_mass", &value.mass);
            put(
                &mut rr.values,
                "included_absolute_mass",
                &value.absolute_mass,
            );
            put(&mut rr.values, "remaining_supplied_mass", &value.remaining);
            put(
                &mut rr.values,
                "included_count",
                &Float::with_val(p, value.count),
            );
            put(
                &mut rr.values,
                "arithmetic_precision_bits",
                &Float::with_val(p, value.arithmetic_precision),
            );
            if let Some(moments) = value.moments {
                for (j, value) in moments.iter().enumerate() {
                    put(
                        &mut rr.values,
                        &format!("weighted_inverse_moment_{}", j + 1),
                        value,
                    );
                }
            } else {
                rr.outcome = "unresolved_denominator".into();
                r.outcome = "partial_unresolved".into();
                if let Some(reason) = value.reason {
                    rr.notes.push(reason);
                }
            }
            rr.notes.push("unprovided infinite tail not bounded; known-zero geometry remains an input; finite arithmetic excludes source construction error".into());
            r.rows.push(rr);
        }
    }
    Ok(r)
}

fn cluster(
    s: &RetainedState,
    o: &ExtensionOptions,
    input: Option<&ExternalResearchInputs>,
) -> Result<ExtendedAnalysis> {
    let mut r = report("spectral_cluster", s, o);
    let Some(i) = input.filter(|i| !i.cluster.is_empty()) else {
        return Ok(missing(
            r,
            "explicit retained cluster vectors and assembly policies required",
        ));
    };
    let p = o.working_precision_bits;
    let modes = i
        .cluster
        .iter()
        .chain(&i.previous_cluster)
        .map(|c| c.n_modes)
        .max()
        .unwrap()
        .max(s.modes);
    let vectors = 1 + i.cluster.len() + i.previous_cluster.len();
    let scratch = (8u64 * (vectors as u64) * (2 * modes as u64 + 1) + 8192)
        * ((u64::from(p) + 4096).div_ceil(8) + 128);
    if o.maximum_working_bytes.is_some_and(|limit| scratch > limit) {
        return Ok(unresolved(
            r,
            "finite cluster scratch estimate exceeds explicit working-byte budget",
        ));
    }
    let current = i
        .cluster
        .iter()
        .map(|c| coeffs(&c.coefficients, c.precision_bits))
        .collect::<Result<Vec<_>>>()?;
    let previous = i
        .previous_cluster
        .iter()
        .map(|c| coeffs(&c.coefficients, c.precision_bits))
        .collect::<Result<Vec<_>>>()?;
    let measured = cluster_math::measure(&s.coefficients, &current, &previous, p)?;
    put(
        &mut r.values,
        "arithmetic_precision_bits",
        &Float::with_val(p, measured.arithmetic_precision),
    );
    for (a, overlap) in measured.overlaps.iter().enumerate() {
        put(&mut r.values, &format!("source_overlap_{a}"), overlap);
        for k in 0..current.len() {
            put(
                &mut r.values,
                &format!("gram_{a}_{k}"),
                &measured.gram[a * current.len() + k],
            );
        }
    }
    if let (Some(leakage), Some(pivot)) = (measured.leakage, measured.minimum_pivot) {
        put(&mut r.values, "source_cluster_leakage_squared", &leakage);
        put(&mut r.values, "minimum_gram_pivot", &pivot);
    } else {
        r.outcome = "rank_or_precision_unresolved".into();
        r.reason = Some("unit-column Gram rank or 4096-bit precision budget unresolved; no exact-rank assertion".into());
    }
    for (a, overlaps) in measured.previous.into_iter().enumerate() {
        let mut rr = row(a + 1, "current_cluster_vector");
        let eigen = Float::with_val(
            p,
            scalar(&i.cluster[a].eigenvalue, i.cluster[a].precision_bits)?,
        );
        put(&mut rr.values, "eigenvalue", &eigen);
        let mut matches = Vec::new();
        for (b, overlap) in overlaps.into_iter().enumerate() {
            put(&mut rr.values, &format!("previous_overlap_{b}"), &overlap);
            matches.push((b, overlap.abs()));
        }
        matches.sort_by(|a, b| b.1.total_cmp(&a.1));
        if let Some((best, overlap)) = matches.first() {
            put(&mut rr.values, "largest_previous_overlap", overlap);
            put(
                &mut rr.values,
                "best_previous_index",
                &Float::with_val(p, *best),
            );
            if matches.len() > 1 {
                let margin = point::sum(&[overlap.clone(), -matches[1].1.clone()], p)?;
                if margin == 0 {
                    rr.notes.push("overlap tie at reported precision; lowest previous index retained; no unique match established".into());
                }
                put(&mut rr.values, "match_margin", &margin);
            }
        }
        rr.notes.push("cross-N zero-padding at common support; original vector signs retained; point overlap matching does not certify branch identity".into());
        r.rows.push(rr);
    }
    if let Some([low, high]) = &i.cluster_boundary_eigenvalues {
        let low = scalar(low, i.precision_bits)?;
        let high = scalar(high, i.precision_bits)?;
        put(
            &mut r.values,
            "declared_boundary_gap",
            &point::sum(&[high, -low], p)?,
        );
    }
    r.convention="unit coefficient norm; same-support Fourier embedding; guarded point nonorthogonal projection; minimum pivot refers to unit-column Gram; source points decoded at own precision; no construction-error or rank certificate".into();
    Ok(r)
}

fn resolution(
    s: &RetainedState,
    roots: Option<&RetainedRoots>,
    o: &ExtensionOptions,
    input: Option<&ExternalResearchInputs>,
) -> Result<ExtendedAnalysis> {
    use super::retained_evidence::{
        finite_math::{abs, decimal},
        transform_math,
    };
    use xc_numerics::mpfr_interval::MpfrInterval as I;
    // Named allowances use outward upper endpoints, while the paired endpoints
    // retain the enclosure of the conditional expression itself.
    fn upper(map: &mut BTreeMap<String, String>, name: &str, x: &I, p: u32) -> Result<()> {
        save_arithmetic_enclosure(map, name, x, p)?;
        map.insert(name.into(), map[&format!("{name}_upper")].clone());
        Ok(())
    }
    let mut r = report("resolution_budget", s, o);
    let p = o.working_precision_bits;
    let reference = input.map(|i| i.reference_jets.as_slice()).unwrap_or(&[]);
    if reference.is_empty() && roots.is_none() {
        return Ok(missing(r, "retained roots or reference ordinates required"));
    }
    let scratch = (8 * s.coefficients.len() as u64 + 256) * (u64::from(p + 4096).div_ceil(8) + 64);
    if o.maximum_working_bytes.is_some_and(|limit| scratch > limit) {
        return Ok(unresolved(
            r,
            "resolution maximum-guard scratch exceeds explicit working-byte budget",
        ));
    }
    let Some(curvature) = transform_math::curvature(&s.cutoff, p)? else {
        return Ok(unresolved(
            r,
            "finite curvature unresolved within 4096 guard bits",
        ));
    };
    upper(&mut r.values, "finite_curvature_expression", &curvature, p)?;
    let point_precision = if reference.is_empty() {
        roots.unwrap().dataset.precision_bits
    } else {
        input.unwrap().precision_bits
    };
    let external_precision = input.map_or(p, |i| i.precision_bits);
    let points: Vec<_> = if reference.is_empty() {
        roots
            .unwrap()
            .dataset
            .points
            .iter()
            .map(|r| (r.ordinal, r.value.clone(), r.source_status.clone()))
            .collect()
    } else {
        reference
            .iter()
            .map(|r| {
                (
                    r.ordinal,
                    Some(r.t.clone()),
                    "external_reference_ordinate".into(),
                )
            })
            .collect()
    };

    let mut qualifying = 0usize;
    let mut contiguous = true;
    let mut previous: Option<usize> = None;
    for (ordinal, t, status) in points {
        let mut rr = row(ordinal, status);
        let Some(t) = t else {
            rr.outcome = "missing_input".into();
            contiguous = false;
            r.rows.push(rr);
            continue;
        };
        let t = scalar(&t, point_precision)?;
        let measured = if reference.is_empty() {
            transform_math::measure_root(s, &t, p)?
        } else {
            transform_math::measure(s, &t, p)?
        };
        let Some(m) = measured else {
            rr.outcome = "cancellation_limited".into();
            rr.notes.push("finite transform enclosure unresolved within 4096 guard bits; all budget claims withheld".into());
            contiguous = false;
            r.rows.push(rr);
            continue;
        };
        let work = m.precision;
        let ti = I::from_float(&t, work)?;
        save_arithmetic_enclosure(&mut rr.values, "t", &ti, p)?;
        put(
            &mut rr.values,
            "arithmetic_precision_bits",
            &Float::with_val(p, work),
        );
        let neighbors = reference
            .iter()
            .filter(|j| j.ordinal.abs_diff(ordinal) == 1)
            .map(|j| Ok((j.ordinal, scalar(&j.t, external_precision)?)))
            .collect::<Result<Vec<_>>>()?;
        let ordered = neighbors.iter().all(|(index, value)| {
            if *index < ordinal {
                value < &t
            } else {
                value > &t
            }
        });
        let spacing = if ordered {
            let mut spacing: Option<I> = None;
            for (_, v) in &neighbors {
                let difference = abs(&I::from_float(v, work)?.sub(&ti))?;
                spacing = Some(if let Some(old) = spacing {
                    I::new(
                        old.lower().clone().min(difference.lower()),
                        old.upper().clone().min(difference.upper()),
                    )?
                } else {
                    difference
                });
            }
            spacing.filter(I::is_strictly_positive)
        } else {
            rr.notes.push("reference neighbors are not strictly ordered at their declared precision; spacing ratios withheld".into());
            None
        };
        save_arithmetic_enclosure(
            &mut rr.values,
            "supplied_adjacent_reference_count",
            &I::from_u64(neighbors.len() as u64, work),
            p,
        )?;
        if let Some(gap) = &spacing {
            save_arithmetic_enclosure(&mut rr.values, "reference_neighbor_spacing", gap, p)?;
            rr.notes.push("spacing uses supplied adjacent reference ordinals; no zeta identification is inferred".into());
        }
        if let Some(j) = reference.iter().find(|j| j.ordinal == ordinal) {
            if let Some(join) = j.matched_root_ordinal {
                if let Some((root, value)) = roots.and_then(|root| {
                    root.dataset
                        .points
                        .iter()
                        .find(|v| v.ordinal == join)
                        .and_then(|v| v.value.as_ref())
                        .map(|v| (root, v))
                }) {
                    let delta =
                        I::from_float(&scalar(value, root.dataset.precision_bits)?, work)?.sub(&ti);
                    save_arithmetic_enclosure(
                        &mut rr.values,
                        "matched_retained_root_ordinal",
                        &I::from_u64(join as u64, work),
                        p,
                    )?;
                    save_arithmetic_enclosure(
                        &mut rr.values,
                        "matched_root_reference_displacement",
                        &delta,
                        p,
                    )?;
                    if let Some(gap) = &spacing {
                        save_arithmetic_enclosure(
                            &mut rr.values,
                            "spacing_normalized_displacement",
                            &delta.div(gap)?,
                            p,
                        )?;
                    }
                    rr.notes.push("root join declared by caller and bound to retained source; not certified zero identification".into());
                } else {
                    rr.notes.push(
                        "declared retained-root join unavailable; displacement unassessed".into(),
                    );
                }
            }
        }
        for (name, value) in [
            ("transform", &m.value),
            ("derivative", &m.derivative),
            ("absolute_value_terms", &m.absolute_terms),
            ("absolute_derivative_terms", &m.absolute_derivative_terms),
        ] {
            save_arithmetic_enclosure(&mut rr.values, name, value, p)?;
        }
        if m.derivative.contains_zero() {
            rr.outcome = "unresolved_derivative".into();
        } else {
            let newton = m.value.neg().div(&m.derivative)?;
            save_arithmetic_enclosure(&mut rr.values, "point_newton_correction", &newton, p)?;
            if let Some(gap) = &spacing {
                save_arithmetic_enclosure(
                    &mut rr.values,
                    "spacing_normalized_newton_correction",
                    &newton.div(gap)?,
                    p,
                )?;
            }
            rr.outcome = if m.value.contains_zero() {
                "cancellation_limited"
            } else {
                "channels_resolved_budget_unassessed"
            }
            .into();
            if let Some(j) = reference.iter().find(|j| j.ordinal == ordinal) {
                if let (Some(ev), Some(ed), Some(h)) = (
                    &j.source_value_error,
                    &j.source_derivative_error,
                    &j.root_separation_radius,
                ) {
                    let ev = I::from_float(&scalar(ev, external_precision)?, work)?;
                    let ed = I::from_float(&scalar(ed, external_precision)?, work)?;
                    let h = I::from_float(&scalar(h, external_precision)?, work)?;
                    let numerator = abs(&m.value)?.add(&ev);
                    let slope = abs(&m.derivative)?.sub(&ed).sub(&m.curvature.mul(&h));
                    for (name, value) in [
                        ("declared_source_value_error", &ev),
                        ("declared_source_derivative_error", &ed),
                        ("declared_radius", &h),
                        ("conditional_slope_margin", &slope),
                        ("conditional_value_numerator", &numerator),
                    ] {
                        save_arithmetic_enclosure(&mut rr.values, name, value, p)?;
                    }
                    if let Some(tail) = &j.tail_value_error {
                        save_arithmetic_enclosure(
                            &mut rr.values,
                            "declared_reference_tail_error",
                            &I::from_float(&scalar(tail, external_precision)?, work)?,
                            p,
                        )?;
                    }
                    if h.is_strictly_positive() && slope.is_strictly_positive() {
                        let bound = numerator.div(&slope)?;
                        upper(
                            &mut rr.values,
                            "conditional_root_distance_allowance",
                            &bound,
                            p,
                        )?;
                        if let Some(gap) = &spacing {
                            upper(
                                &mut rr.values,
                                "spacing_normalized_conditional_allowance",
                                &bound.div(gap)?,
                                p,
                            )?;
                        }
                        let target = h.mul(&decimal(&o.relative_tolerance, work)?);
                        save_arithmetic_enclosure(
                            &mut rr.values,
                            "conditional_budget_target",
                            &target,
                            p,
                        )?;
                        // Include the final outward decimal serialization in the
                        // decision, including exact-equality boundary cases.
                        let advertised =
                            decimal(&rr.values["conditional_root_distance_allowance"], work)?;
                        let advertised_target =
                            decimal(&rr.values["conditional_budget_target_lower"], work)?;
                        rr.outcome = if advertised.upper() <= advertised_target.lower() {
                            "conditional_budget_met"
                        } else if bound.lower() > target.upper() {
                            "conditional_budget_not_met"
                        } else {
                            "conditional_budget_unresolved"
                        }
                        .into();
                    } else {
                        rr.outcome = "conditional_budget_unresolved".into();
                    }
                    rr.notes.push("conditional on supplied absolute source errors and an isolated root within the declared radius with this curvature bound; those target hypotheses are not certified here".into());
                }
            }
        }
        if previous.is_none() && ordinal != 1
            || previous.is_some_and(|x| x.checked_add(1) != Some(ordinal))
            || rr.outcome != "conditional_budget_met"
        {
            contiguous = false;
        }
        if contiguous {
            qualifying = ordinal;
        }
        previous = Some(ordinal);
        r.rows.push(rr);
    }
    put(
        &mut r.values,
        "conditional_contiguous_prefix",
        &Float::with_val(p, qualifying),
    );
    save_arithmetic_enclosure(
        &mut r.values,
        "relative_tolerance",
        &decimal(&o.relative_tolerance, p)?,
        p,
    )?;
    r.convention="unit_L2_dx finite transform arithmetic enclosures; exact decimal cutoff and original stored points; retained-root ordinals are not zeta ordinals; conditional finite-source radius budget; not minimum-N law".into();
    Ok(r)
}

fn allowance(
    s: &RetainedState,
    o: &ExtensionOptions,
    input: Option<&ExternalResearchInputs>,
) -> Result<ExtendedAnalysis> {
    let mut r = report("energy_allowance", s, o);
    let Some(a) = input.and_then(|i| i.energy_allowance.as_ref()) else {
        return Ok(missing(
            r,
            "declared block bounds and their hypothesis record required",
        ));
    };
    let p = o.working_precision_bits;
    if o.maximum_working_bytes
        .is_some_and(|limit| 128 * (u64::from(p + 4096).div_ceil(8) + 64) > limit)
    {
        return Ok(unresolved(
            r,
            "allowance maximum-guard scratch exceeds explicit working-byte budget",
        ));
    }
    let Some(measured) = allowance_math::measure(a, input.unwrap().precision_bits, p)? else {
        return Ok(unresolved(
            r,
            "allowance arithmetic width unresolved within 4096 guard bits",
        ));
    };
    for (name, value) in &measured.values {
        save_arithmetic_enclosure(&mut r.values, name, value, p)?;
    }
    put(
        &mut r.values,
        "arithmetic_precision_bits",
        &Float::with_val(p, measured.precision),
    );
    if !measured.valid_margin {
        r.outcome = "sufficient_bound_unavailable".into();
        r.reason=Some("high block lower bound is not above trial upper energy at declared input precision; no division performed".into());
    } else {
        r.outcome = "conditional_bound_expression".into();
        r.reason=Some(if measured.zero_trial {
            "trial energy is zero; relative scale comparison unavailable; no division by trial energy performed"
        } else {match measured.below {
            Some(true)=>"allowance is smaller than the trial energy magnitude; scale comparison only, not a relative eigenvalue error certificate",
            Some(false)=>"non-informative at the trial energy scale: allowance is at least the trial energy magnitude; not evidence of relative energy accuracy",
            None=>"strict comparison with trial energy magnitude remains unresolved: arithmetic enclosures overlap; below-scale flag is not established, not a claim of greater-than or equality",
        }}.into());
    }

    r.convention="conditional H^2/(mu-U) and H/(mu-U); externally declared block bounds and hypotheses; not a certificate".into();
    Ok(r)
}
/// Capture the admitted external numeric input itself, retaining no target formula.
pub fn capture_external_source(
    s: &RetainedState,
    i: &ExternalResearchInputs,
    cache: &ArtifactCacheContext<'_>,
) -> Result<ArtifactExecutionCacheResult<ResearchRecord<ExternalResearchInputs>>> {
    capture_external_source_with_parents(s, i, &[], cache)
}
fn capture_external_source_with_parents(
    s: &RetainedState,
    i: &ExternalResearchInputs,
    parents: &[ArtifactManifest],
    cache: &ArtifactCacheContext<'_>,
) -> Result<ArtifactExecutionCacheResult<ResearchRecord<ExternalResearchInputs>>> {
    i.matches(s)?;
    managed(
        INPUT_KIND,
        json!({"input_digest":ContentDigest::sha256(&serde_json::to_vec(i)?)}),
        &[std::slice::from_ref(&s.manifest), parents].concat(),
        cache,
        || Ok(i.clone()),
        |r| {
            r.matches(s)?;
            if r != i {
                bail!("external research source mismatch");
            }
            Ok(())
        },
    )
}
/// A finite diagnostic computed from immutable admitted primary sources and
/// explicitly supplied optional reference/operator data. Missing inputs produce
/// a qualified report; the live outcome adapter maps that report to Missing.
#[allow(clippy::too_many_arguments)] // Keep independently admitted sources explicit.
pub fn capture_extended(
    id: &str,
    s: &RetainedState,
    m: Option<&RetainedMatrix<'_>>,
    roots: Option<&RetainedRoots>,
    input: Option<&ExternalResearchInputs>,
    options: &ExtensionOptions,
    parent_manifests: &[ArtifactManifest],
    cache: &ArtifactCacheContext<'_>,
) -> Result<ArtifactExecutionCacheResult<ResearchRecord<ExtendedAnalysis>>> {
    let kind =
        artifact_kind(id).ok_or_else(|| anyhow::anyhow!("unknown extended research diagnostic"))?;
    let resource_policy = super::capture_runtime::CaptureResourcePolicy::from_environment()?;
    let mut effective_options = options.clone();
    effective_options.maximum_working_bytes = Some(
        options
            .maximum_working_bytes
            .unwrap_or(resource_policy.maximum_working_bytes),
    );
    let options = &effective_options;
    options.validate(s)?;
    if let Some(i) = input {
        i.matches(s)?;
        if options.working_precision_bits < i.precision_bits {
            bail!("working precision below external source precision");
        }
    }
    let mut sources = vec![s.manifest.clone()];
    if let Some(m) = m {
        m.match_state(s)?;
        sources.extend(matrix_ancestry(s, m, parent_manifests, cache)?);
        sources.push(m.manifest.clone());
    }
    if let Some(r) = roots {
        if options.working_precision_bits < r.dataset.precision_bits {
            bail!("working precision below retained root precision");
        }
        if !xc_cache::manifest_depends_on(&r.secular_manifest, &s.manifest)? {
            bail!("roots belong to another state");
        }
        sources.extend([r.manifest.clone(), r.secular_manifest.clone()]);
    }
    let input_digest = if let Some(i) = input {
        let result = capture_external_source_with_parents(s, i, parent_manifests, cache)?;
        let manifest = result.produced_manifest.or(result.reused_manifest);
        if let Some(m) = manifest {
            sources.push(m);
        }
        Some(ContentDigest::sha256(&serde_json::to_vec(i)?))
    } else {
        None
    };
    let directional_source = if let ("root_transport", Some(_), Some(r)) = (id, m, roots) {
        let mut full = options.clone();
        full.maximum_directional_rows = r.dataset.points.len();
        let result = capture_extended(
            "directional_response",
            s,
            m,
            roots,
            input,
            &full,
            parent_manifests,
            cache,
        )?;
        if let Some(manifest) = result
            .produced_manifest
            .as_ref()
            .or(result.reused_manifest.as_ref())
        {
            sources.push(manifest.clone());
        }
        Some(result.value.data)
    } else {
        None
    };
    let tail_rows =
        input.map_or(0, |i| {
            let groups = i
                .atoms
                .iter()
                .map(|a| (&a.family, &a.partition))
                .collect::<BTreeSet<_>>()
                .len();
            let checkpoints = if i.tail_checkpoints.is_empty() {
                usize::BITS as usize - i.atoms.len().max(1).leading_zeros() as usize + 1
            } else {
                i.tail_checkpoints.len()
            };
            groups.saturating_mul(checkpoints.saturating_add(
                super::atom_research::policy(input).map_or(0, |a| a.evaluations.len()),
            ))
        });
    let estimated_rows = match id {
        "compactness" => options.exponential_rates.len(),
        "weighted_reference_projection" | "energy_allowance" => 0,
        "signed_transform" => input.map_or(0, |i| i.reference_jets.len()),
        "arithmetic_energy" => input.map_or(0, ExternalResearchInputs::arithmetic_component_count),
        "directional_response" => roots.map_or(0, |r| r.dataset.points.len()),
        "weighted_tail" => tail_rows,
        "spectral_cluster" => input.map_or(0, |i| i.cluster.len()),
        "resolution_budget" => input.filter(|i| !i.reference_jets.is_empty()).map_or_else(
            || roots.map_or(0, |r| r.dataset.points.len()),
            |i| i.reference_jets.len(),
        ),
        "complex_transform" => 70 + 5 * roots.map_or(0, |r| r.dataset.points.len()),
        "root_transport" | "observable_budget" => roots.map_or(0, |r| r.dataset.points.len()),
        "finite_section_transfer" => s.modes + 2,
        "operator_cluster" => 1024,
        "capture_preflight" => 11,
        "consistency" => {
            super::research_completion::completion(input).map_or(0, |c| c.independent_actions.len())
        }
        "configuration_comparison" => {
            super::research_completion::completion(input).map_or(0, |c| c.comparisons.len())
        }
        "band_reconstruction" => super::research_completion::completion(input)
            .and_then(|c| c.band.as_ref())
            .map_or(0, |b| {
                b.degree.saturating_add(
                    super::atom_research::policy(input).map_or(0, |a| a.cutoffs.len()),
                )
            }),
        "transform_enclosure" => super::research_completion::completion(input)
            .and_then(|c| c.contour.as_ref())
            .map_or(4096, |c| c.maximum_segments)
            .saturating_add(roots.map_or(0, |r| r.dataset.points.len())),
        "tail_operator" => input
            .and_then(|i| i.run_once.as_ref())
            .and_then(|i| i.tail_form.as_ref())
            .map_or_else(
                || {
                    super::atom_research::policy(input)
                        .and_then(|a| a.tail_recipe.as_ref())
                        .map_or(0, |r| r.basis_polynomials.len())
                },
                |f| f.dimension,
            )
            .saturating_add(super::atom_research::policy(input).map_or(0, |a| a.cutoffs.len())),
        _ => 0,
    };
    let budget_exceeded = (id == "weighted_tail" && tail_rows > options.maximum_rows)
        || estimated_rows > options.maximum_rows
        || estimated_rows as u64
            * (8192
                + (if id == "resolution_budget" { 96 } else { 64 })
                    * (u64::from(options.working_precision_bits) / 3 + 32))
            > options.maximum_estimated_output_bytes;

    let result = managed(
        kind,
        {
            let mut request = json!({"semantics":if id == "transform_enclosure" { "extended-retained-diagnostics-v5-minus-fourier-source-error-hull" } else if matches!(id,"band_reconstruction"|"tail_operator"|"weighted_tail") { "extended-retained-diagnostics-v4" } else if id == "resolution_budget" { "extended-retained-diagnostics-v3" } else { "extended-retained-diagnostics-v2" },"expected_rows":if matches!(id,"transform_enclosure"|"operator_cluster"|"finite_section_transfer"|"configuration_comparison"|"weighted_tail"|"band_reconstruction"|"tail_operator") { None } else { Some(estimated_rows) },"diagnostic":id,"options":options,"external_input_digest":input_digest,"state_selection_policy":s.selection_policy,"state_manifest_tags":s.manifest.tags});
            if id == "configuration_comparison" {
                request["duplicate_coordinate_policy"] = json!("all_ambiguous_members_withheld_v2");
            }
            if matches!(id, "observable_budget" | "tail_operator") {
                request["declared_error_semantics"] = json!("exact_decimal_upper_bound_v2");
            }
            if id == "band_reconstruction" {
                request["ladder_positivity_policy"] =
                    json!("all_required_recurrence_steps_including_failed_v2");
            }
            request["source_unit_arithmetic"] = json!("binary_scaled_hypot_checked_range_v2");
            request["resource_admission"] = json!("resolved_working_bytes_v1");
            if id == "complex_transform" {
                request["maximum_parallel_rows"] = json!(resource_policy.root_block_rows);
                request["workspace_admission"] = json!("configured_row_block_bound_v1");
            }
            if id == "band_reconstruction" {
                request["exact_contraction_admission"] = json!("all_block_workspace_bound_v1");
            }
            if id == "weighted_reference_projection" {
                request["weighted_profile_arithmetic"] =
                    json!("stored_points_combined_difference_interval_gram_unresolved_v2");
                request["maximum_weighted_profile_guard_bits"] = json!(4096);
                request["weighted_profile_output"] =
                    json!("midpoint_with_outward_decimal_enclosures_v1");
            }
            if id == "directional_response" {
                request["directional_arithmetic"] =
                    json!("stored_points_projected_resolvent_displacement_checked_v2");
                request["maximum_directional_guard_bits"] = json!(4096);
                request["directional_output"] =
                    json!("midpoints_with_outward_decimal_enclosures_v1");
                request["root_point_precision"] = json!("declared_payload_precision_v1");
            }
            if id == "arithmetic_energy" {
                request["energy_arithmetic"] =
                    json!("stored_points_scaled_quadratic_intervals_deficit_kind_v2");
                request["component_selection"] =
                    json!("explicit_operators_else_compact_actions_v1");
                request["maximum_energy_guard_bits"] = json!(4096);
                request["energy_output"] = json!("midpoints_with_outward_decimal_enclosures_v1");
            }
            if id == "signed_transform" {
                request["signed_channel_arithmetic"] =
                    json!("exact_cutoff_center_declared_points_checked_channels_v3");
            }
            if id == "spectral_cluster" {
                request["cluster_arithmetic"] = json!("stored_points_scaled_unit_checked_gram_v3");
                request["maximum_cluster_guard_bits"] = json!(4096);
                request["cluster_precision_policy"] = json!("unit_column_pivot_proxy_v1");
            }
            if id == "weighted_tail" {
                request["atom_arithmetic"] = json!("stored_points_exact_mass_directed_moments_v2");
                request["atom_coordinate_serialization"] = json!("promoted_source_point_v1");
                request["maximum_atom_guard_bits"] = json!(4096);
                request["maximum_atom_exponent_span_bits"] = json!(1_000_000);
            }
            if id == "compactness" {
                request["compactness_arithmetic"] =
                    json!("directed_enclosure_agreed_rounding_or_unresolved_v2");
                request["maximum_additional_guard_bits"] = json!(4096);
            }
            if id == "band_reconstruction" {
                request["polynomial_band_arithmetic"] =
                    json!("stored_polynomial_exact_newton_inverse_moments_v1");
                request["polynomial_root_output"] =
                    json!("outward_root_bounds_and_safe_midpoints_v1");
                request["polynomial_root_window"] =
                    json!("common_binary_scale_exact_rational_cauchy_v2");
                request["maximum_polynomial_band_exact_bits"] = json!(8_000_000);
                request["signed_band_arithmetic"] =
                    json!("declared_points_normalized_recurrence_exact_contractions_v1");
                request["maximum_signed_band_exact_bits"] = json!(8_000_000);
                request["signed_band_inverse_arithmetic"] =
                    json!("relative_zero_guard_scaled_directed_sums_v1");
                request["maximum_signed_band_inverse_guard_bits"] = json!(4096);
                request["basis_disk_budget_bytes"] = json!(super::band_runtime::disk_budget()?);
                request["checkpoint_block_budget_bytes"] = json!(
                    super::capture_runtime::CaptureResourcePolicy::from_environment()?
                        .maximum_checkpoint_bytes
                );
            }
            // Feature-dependent absence and polynomial isolation must not share
            // a cache identity with an Arb-enabled calculation, including legacy
            // records that did not state the available numerical backend.
            if matches!(id, "transform_enclosure" | "band_reconstruction") {
                request["arb_available"] = json!(cfg!(feature = "arb"));
            }
            if id == "transform_enclosure" {
                request["enclosure_fourier_semantics"] =
                    json!(super::transform_enclosure::FOURIER_SEMANTICS);
                request["enclosure_algorithm"] =
                    json!("centered-taylor-48-integral-remainder-minus-fourier-v4-stored-points");
                request["enclosure_point_precision"] =
                    json!("declared_payload_and_external_precision_v1");
                request["enclosure_decimal_output"] = json!("outward_endpoints_v1");
            }
            if matches!(id, "tail_operator" | "band_reconstruction") {
                request["tail_model_checkpoint_arithmetic"] =
                    json!("finite-tail-model-original-matrix-dense-source-recovery-v8");
                request["tail_model_householder_arithmetic"] =
                    json!(xc_numerics::eigen::STABLE_HOUSEHOLDER_SEMANTICS);
                request["tail_model_qr_arithmetic"] =
                    json!(xc_numerics::eigen::TRIDIAG_QR_SEMANTICS);
                request["tail_model_vector_recovery_arithmetic"] =
                    json!(xc_numerics::eigen::DENSE_EIGENVECTOR_SEMANTICS);
                request["tail_model_arithmetic"] =
                    json!("declared_points_exact_dyadic_recipe_forms_v2");
                request["maximum_tail_form_exact_bits"] = json!(8_000_000);
            }
            if id == "root_transport" {
                request["transport_arithmetic"] =
                    json!("exact_cutoff_stored_points_directional_intervals_v1");
                request["transport_output"] = json!("midpoints_with_outward_decimal_enclosures_v1");
                request["maximum_transport_guard_bits"] = json!(4096);
            }
            if id == "tail_operator" {
                request["model_linear_algebra_arithmetic"] =
                    json!("exact_stored_dot_product_stages_and_tail_bound_v0.15.2-v2");
                request["l2_normalization_arithmetic"] =
                    json!(xc_numerics::linalg::L2_NORMALIZATION_ARITHMETIC_V2);
            }
            if matches!(
                id,
                "signed_transform"
                    | "resolution_budget"
                    | "observable_budget"
                    | "configuration_comparison"
            ) {
                request["transform_arithmetic"] =
                    json!("exact_cutoff_stored_points_directed_sinc_v1");
                request["maximum_transform_guard_bits"] = json!(4096);
            }
            if id == "resolution_budget" {
                request["resolution_arithmetic"] =
                    json!("exact_decimal_tolerance_original_point_conditional_distance_v2");
                request["resolution_output"] =
                    json!("outward_allowance_upper_endpoints_with_expression_enclosures_v1");
            }
            if id == "finite_section_transfer" {
                request["finite_transfer_arithmetic"] =
                    json!("original_points_scaled_prefix_shifted_residual_v1");
                request["finite_transfer_output"] =
                    json!("midpoints_with_outward_decimal_enclosures_v1");
                request["maximum_finite_transfer_guard_bits"] = json!(4096);
            }
            if id == "consistency" {
                request["consistency_arithmetic"] =
                    json!("original_points_scaled_action_difference_v1");
                request["consistency_output"] =
                    json!("midpoints_with_outward_decimal_enclosures_v1");
                request["maximum_consistency_guard_bits"] = json!(4096);
            }
            if id == "complex_transform" {
                request["complex_fourier_semantics"] =
                    json!(super::transform_enclosure::FOURIER_SEMANTICS);
                request["complex_arithmetic"] =
                    json!("exact_cutoff_original_points_directed_entire_minus_sinc_v2");
                request["complex_point_construction"] =
                    json!("original_probes_exact_affine_contour_minus_fourier_v2");
                request["complex_output"] = json!("midpoints_with_outward_decimal_enclosures_v1");
                request["maximum_complex_guard_bits"] = json!(4096);
                request["complex_root_point_precision"] = json!("declared_payload_precision_v1");
            }
            if id == "operator_cluster" {
                request["cluster_operator_arithmetic"] =
                    json!("original_points_shift_before_interval_projection_lu_v1");
                request["cluster_operator_output"] =
                    json!("midpoints_with_outward_decimal_enclosures_v1");
                request["maximum_cluster_operator_guard_bits"] = json!(4096);

                request["cluster_basis_arithmetic"] =
                    json!("original_points_unit_columns_before_rank_threshold_v1");
            }
            if id == "observable_budget" {
                request["observation_arithmetic"] =
                    json!("original_points_directed_l2_transport_v1");
                request["observation_output"] =
                    json!("allowance_upper_margin_lower_with_expression_enclosures_v1");
            }
            if id == "configuration_comparison" {
                request["comparison_arithmetic"] =
                    json!("original_points_scaled_overlap_shifted_residual_v1");
                request["comparison_output"] =
                    json!("midpoints_with_outward_decimal_enclosures_v1");
                request["maximum_comparison_guard_bits"] = json!(4096);
            }
            if id == "energy_allowance" {
                request["allowance_interpretation"] = json!("trial-energy-scale-v1");
                request["allowance_arithmetic"] =
                    json!("declared_points_separate_binary_scales_intervals_v1");
                request["maximum_allowance_guard_bits"] = json!(4096);
                request["allowance_output"] = json!("midpoints_with_outward_decimal_enclosures_v1");
            }
            request
        },
        &sources,
        cache,
        || {
            let _stage = super::capture_runtime::Stage::new(format!("{id} compute"));
            if budget_exceeded {
                return Ok(unresolved(
                    report(id, s, options),
                    "row/output resource budget exceeded; retry this diagnostic with explicit limits",
                ));
            }
            match id {
                "compactness" => compactness(s, options),
                "weighted_reference_projection" => weighted_projection(s, options, input),
                "signed_transform" => signed_transforms(s, options, input),
                "arithmetic_energy" => arithmetic_energy(s, m, options, input),
                "directional_response" => directional(s, m, roots, options, input),
                "weighted_tail" => super::atom_research::weighted_report(s, options, input),
                "spectral_cluster" => cluster(s, options, input),
                "resolution_budget" => resolution(s, roots, options, input),
                "energy_allowance" => allowance(s, options, input),
                _ if super::research_completion::DIAGNOSTICS.contains(&id) => {
                    super::research_completion::analyze(id, s, m, roots, options, input)
                }
                _ => super::convergence_capture::analyze(
                    id,
                    s,
                    m,
                    roots,
                    options,
                    input,
                    directional_source.as_ref(),
                ),
            }
        },
        |r| {
            if (r.outcome == "certified_finite_enclosure" && id != "transform_enclosure")
                || r.diagnostic != id
                || r.lambda_squared != s.cutoff
                || r.n_modes != s.modes
                || r.source_precision_bits != s.precision
                || r.working_precision_bits != options.working_precision_bits
                || r.assurance
                    != if id == "transform_enclosure" {
                        "finite_retained_function_enclosures; source_scope_explicit; no_infinite_limit_claim"
                    } else if id == "directional_response" {
                        DIRECTIONAL_ASSURANCE
                    } else if id == "root_transport" {
                        TRANSPORT_ASSURANCE
                    } else if id == "complex_transform" {
                        COMPLEX_ASSURANCE
                    } else if id == "operator_cluster" {
                        OPERATOR_CLUSTER_ASSURANCE
                    } else if id == "finite_section_transfer" {
                        TRANSFER_ASSURANCE
                    } else if id == "consistency" {
                        CONSISTENCY_ASSURANCE
                    } else if id == "observable_budget" {
                        OBSERVATION_ASSURANCE
                    } else if id == "configuration_comparison" {
                        COMPARISON_ASSURANCE
                    } else if id == "resolution_budget" {
                        RESOLUTION_ASSURANCE
                    } else if id == "energy_allowance" {
                        ALLOWANCE_ASSURANCE
                    } else if id == "arithmetic_energy" {
                        ENERGY_ASSURANCE
                    } else if id == "weighted_reference_projection" {
                        WEIGHTED_PROFILE_ASSURANCE
                    } else {
                        ASSURANCE
                    }
                || r.convention.is_empty()
                || ![
                    "point_measurement",
                    "certified_finite_enclosure",
                    "partial_unresolved",
                    "unresolved",
                    "missing_input",
                    "rank_or_precision_unresolved",
                    "conditional_bound_expression",
                    "sufficient_bound_unavailable",
                ]
                .contains(&r.outcome.as_str())
                || r.rows.len() > options.maximum_rows
            {
                bail!("invalid extended research report identity/status");
            }
            if [
                "missing_input",
                "unresolved",
                "rank_or_precision_unresolved",
                "sufficient_bound_unavailable",
            ]
            .contains(&r.outcome.as_str())
                && r.reason.as_ref().is_none_or(|v| v.is_empty())
            {
                bail!("qualified absence requires a reason");
            }
            for v in r
                .values
                .values()
                .chain(r.rows.iter().flat_map(|r| r.values.values()))
            {
                scalar(v, options.working_precision_bits)?;
            }
            if id == "root_transport" {
                for row in &r.rows {
                    let fields = [
                        "component_forcing_sum",
                        "absolute_component_forcing_sum",
                        "forcing_closure_defect",
                        "support_motion",
                        "operator_motion",
                        "conditional_total_physical_velocity",
                        "retained_secular_pole_motion",
                        "retained_total_velocity",
                        "retained_transport_additivity_defect",
                    ];
                    if fields.iter().any(|name| row.values.contains_key(*name)) {
                        let work = scalar(
                            row.values
                                .get("transport_arithmetic_precision_bits")
                                .ok_or_else(|| {
                                    anyhow::anyhow!("missing transport arithmetic precision")
                                })?,
                            options.working_precision_bits,
                        )?;
                        if work < options.working_precision_bits + 64
                            || work > options.working_precision_bits + 4096
                        {
                            bail!("invalid transport arithmetic precision");
                        }
                    }
                    for name in fields {
                        if let Some(value) = row.values.get(name) {
                            let lo = scalar(
                                row.values.get(&format!("{name}_lower")).ok_or_else(|| {
                                    anyhow::anyhow!("missing transport lower endpoint")
                                })?,
                                options.working_precision_bits,
                            )?;
                            let hi = scalar(
                                row.values.get(&format!("{name}_upper")).ok_or_else(|| {
                                    anyhow::anyhow!("missing transport upper endpoint")
                                })?,
                                options.working_precision_bits,
                            )?;
                            let mid = scalar(value, options.working_precision_bits)?;
                            if lo > mid || mid > hi {
                                bail!("invalid transport enclosure");
                            }
                        }
                    }
                }
            }
            for row in &r.rows {
                if (row.outcome == "certified_finite_enclosure" && id != "transform_enclosure")
                    || row.ordinal == 0
                    || row.label.is_empty()
                    || ![
                        "point_measurement",
                        "certified_finite_enclosure",
                        "missing_input",
                        "budget_limited",
                        "carrier_or_unresolved",
                        "unresolved_denominator",
                        "cancellation_limited",
                        "unresolved_derivative",
                        "channels_resolved_budget_unassessed",
                        "conditional_budget_met",
                        "conditional_budget_not_met",
                        "conditional_budget_unresolved",
                    ]
                    .contains(&row.outcome.as_str())
                {
                    bail!("invalid extended research row");
                }
            }
            if id == "complex_transform"
                && !["missing_input", "unresolved"].contains(&r.outcome.as_str())
            {
                let p = options.working_precision_bits;
                let check = |values: &BTreeMap<String, String>| -> Result<()> {
                    let used = scalar(
                        values.get("arithmetic_precision_bits").ok_or_else(|| {
                            anyhow::anyhow!("missing complex arithmetic precision")
                        })?,
                        p,
                    )?;
                    if used < p + 64 || used > p + 4096 || !used.is_integer() {
                        bail!("invalid complex arithmetic precision");
                    }
                    for (name, value) in values {
                        if name == "arithmetic_precision_bits"
                            || name.ends_with("_lower")
                            || name.ends_with("_upper")
                        {
                            continue;
                        }
                        let lo = values
                            .get(&format!("{name}_lower"))
                            .ok_or_else(|| anyhow::anyhow!("missing complex lower bound"))?;
                        let hi = values
                            .get(&format!("{name}_upper"))
                            .ok_or_else(|| anyhow::anyhow!("missing complex upper bound"))?;
                        let point = scalar(value, p)?;
                        let low =
                            Float::with_val_round(p, Float::parse(lo)?, rug::float::Round::Up).0;
                        let high =
                            Float::with_val_round(p, Float::parse(hi)?, rug::float::Round::Down).0;
                        if low > point || high < point {
                            bail!("complex point outside enclosure");
                        }
                    }
                    Ok(())
                };
                check(&r.values)?;
                if !r.values.contains_key("normalization_anchor") {
                    bail!("complex normalization anchor missing");
                }
                let count = roots.map_or(0, |roots| roots.dataset.points.len());
                let probes = 5 * (count + 1);
                if r.rows.len() != probes + 65 {
                    bail!("complex retained-ordinal or contour row count mismatch");
                }
                for (index, row) in r.rows.iter().enumerate() {
                    check(&row.values)?;
                    if row.ordinal != index + 1
                        || row.label
                            != if index < probes {
                                "complex_transform_sample"
                            } else {
                                "contour_sample"
                            }
                    {
                        bail!("complex row order or label mismatch");
                    }
                    let ordinal = if index >= probes || index < 5 {
                        0
                    } else {
                        roots.unwrap().dataset.points[index / 5 - 1].ordinal
                    };
                    if scalar(
                        row.values
                            .get("input_ordinal")
                            .ok_or_else(|| anyhow::anyhow!("missing complex input ordinal"))?,
                        p,
                    )? != ordinal
                    {
                        bail!("complex input ordinal mismatch");
                    }
                    if index < probes {
                        if scalar(
                            row.values.get("z_im").ok_or_else(|| {
                                anyhow::anyhow!("missing complex imaginary coordinate")
                            })?,
                            p,
                        )? != Float::with_val(p, [-4, -1, 0, 1, 4][index % 5]) / 4u32
                        {
                            bail!("complex sample offset mismatch");
                        }
                        let t = if index < 5 {
                            Some(Float::with_val(p, 0))
                        } else {
                            let roots = roots.unwrap();
                            roots.dataset.points[index / 5 - 1]
                                .value
                                .as_ref()
                                .map(|value| scalar(value, roots.dataset.precision_bits))
                                .transpose()?
                        };
                        if let Some(t) = t {
                            if scalar(
                                row.values.get("z_re").ok_or_else(|| {
                                    anyhow::anyhow!("missing complex real coordinate")
                                })?,
                                p,
                            )? != t
                            {
                                bail!("complex original root point mismatch");
                            }
                            if row.outcome == "missing_input" {
                                bail!("complex present ordinate marked missing");
                            }
                        } else {
                            if row.outcome != "missing_input"
                                || row.values.contains_key("z_re")
                                || row.values.contains_key("value_re")
                            {
                                bail!("complex missing ordinate contract mismatch");
                            }
                            continue;
                        }
                    }
                    if row.outcome == "cancellation_limited" {
                        if row.values.contains_key("value_re") || row.notes.is_empty() {
                            bail!("complex unresolved arithmetic contract mismatch");
                        }
                        continue;
                    }
                    for name in [
                        "z_re",
                        "z_im",
                        "value_re",
                        "value_im",
                        "derivative_re",
                        "derivative_im",
                        "sum_absolute_terms",
                        "normalization_denominator_resolved",
                        "log_derivative_denominator_resolved",
                    ] {
                        if !row.values.contains_key(name) {
                            bail!("complex core measurement missing");
                        }
                    }
                    let mut complete = true;
                    for (flag, name) in [
                        ("normalization_denominator_resolved", "normalized"),
                        ("log_derivative_denominator_resolved", "log_derivative"),
                    ] {
                        let flag = scalar(&row.values[flag], p)?;
                        if flag != 0 && flag != 1 {
                            bail!("invalid complex ratio flag");
                        }
                        let present = flag == 1;
                        complete &= present;
                        for part in ["re", "im"] {
                            if row.values.contains_key(&format!("{name}_{part}")) != present {
                                bail!("complex ratio presence mismatch");
                            }
                        }
                    }
                    if row.outcome
                        != if complete {
                            "point_measurement"
                        } else {
                            "unresolved_denominator"
                        }
                    {
                        bail!("complex ratio outcome mismatch");
                    }
                }
                let a = &r.rows[probes].values;
                let b = &r.rows[probes + 64].values;
                for name in [
                    "z_re",
                    "z_im",
                    "z_re_lower",
                    "z_re_upper",
                    "z_im_lower",
                    "z_im_upper",
                ] {
                    if a.get(name) != b.get(name) {
                        bail!("complex contour is not closed");
                    }
                }
                let partial = r.rows.iter().any(|row| row.outcome != "point_measurement");
                if r.outcome
                    != if partial {
                        "partial_unresolved"
                    } else {
                        "point_measurement"
                    }
                    || (partial && r.reason.as_ref().is_none_or(|v| v.is_empty()))
                {
                    bail!("complex aggregate completion mismatch");
                }
            }
            if id == "operator_cluster"
                && !["missing_input", "unresolved"].contains(&r.outcome.as_str())
            {
                let p = options.working_precision_bits;
                let check = |values: &BTreeMap<String, String>| -> Result<()> {
                    let used = scalar(
                        values.get("arithmetic_precision_bits").ok_or_else(|| {
                            anyhow::anyhow!("missing cluster arithmetic precision")
                        })?,
                        p,
                    )?;
                    if used < p + 64 || used > p + 4096 || !used.is_integer() {
                        bail!("invalid cluster arithmetic precision");
                    }
                    for (name, value) in values {
                        if name == "arithmetic_precision_bits"
                            || name.ends_with("_lower")
                            || name.ends_with("_upper")
                        {
                            continue;
                        }
                        let lo = values
                            .get(&format!("{name}_lower"))
                            .ok_or_else(|| anyhow::anyhow!("missing cluster lower bound"))?;
                        let hi = values
                            .get(&format!("{name}_upper"))
                            .ok_or_else(|| anyhow::anyhow!("missing cluster upper bound"))?;
                        let point = scalar(value, p)?;
                        let low =
                            Float::with_val_round(p, Float::parse(lo)?, rug::float::Round::Up).0;
                        let high =
                            Float::with_val_round(p, Float::parse(hi)?, rug::float::Round::Down).0;
                        if low > point || high < point {
                            bail!("cluster point outside enclosure");
                        }
                    }
                    Ok(())
                };
                check(&r.values)?;
                for name in [
                    "subspace_dimension",
                    "retained_energy_shift",
                    "source_leakage_squared",
                    "estimated_factorization_workspace_bytes",
                    "column_selection_threshold",
                    "discarded_reference_columns",
                ] {
                    if !r.values.contains_key(name) {
                        bail!("cluster report core field missing");
                    }
                }
                let b = scalar(&r.values["subspace_dimension"], p)?
                    .to_integer()
                    .and_then(|v| v.to_usize())
                    .ok_or_else(|| anyhow::anyhow!("invalid cluster dimension"))?;
                if b == 0
                    || b > s.coefficients.len()
                    || scalar(&r.values["subspace_dimension"], p)? != b
                    || r.rows.len() != b * b
                {
                    bail!("cluster row shape mismatch");
                }
                for index in 0..b {
                    if !r
                        .values
                        .contains_key(&format!("selected_input_column_{index}"))
                    {
                        bail!("cluster selected-column identity missing");
                    }
                }
                for (k, row) in r.rows.iter().enumerate() {
                    check(&row.values)?;
                    for name in ["row", "column", "compressed_operator", "coupling_gram"] {
                        if !row.values.contains_key(name) {
                            bail!("cluster row core field missing");
                        }
                    }
                    if row.ordinal != k + 1
                        || scalar(&row.values["row"], p)? != k / b
                        || scalar(&row.values["column"], p)? != k % b
                    {
                        bail!("cluster row coordinates mismatch");
                    }
                    let complete = r.outcome == "point_measurement";
                    if (complete && row.outcome != "point_measurement")
                        || (!complete && row.outcome != "unresolved_denominator")
                    {
                        bail!("cluster row completion mismatch");
                    }
                    for name in [
                        "signed_complement_feedback",
                        "effective_operator",
                        "solve_relative_residual",
                    ] {
                        if row.values.contains_key(name) != complete {
                            bail!("cluster feedback completion mismatch");
                        }
                    }
                }
            }
            if matches!(id, "finite_section_transfer" | "consistency")
                && !["missing_input", "unresolved"].contains(&r.outcome.as_str())
            {
                let p = options.working_precision_bits;
                let check = |values: &BTreeMap<String, String>| -> Result<()> {
                    let used = scalar(
                        values.get("arithmetic_precision_bits").ok_or_else(|| {
                            anyhow::anyhow!("missing transfer/consistency precision")
                        })?,
                        p,
                    )?;
                    if used < p + 64 || used > p + 4096 || !used.is_integer() {
                        bail!("invalid transfer/consistency precision");
                    }
                    for (name, value) in values {
                        if name == "arithmetic_precision_bits"
                            || name.ends_with("_lower")
                            || name.ends_with("_upper")
                        {
                            continue;
                        }
                        let lo = values.get(&format!("{name}_lower")).ok_or_else(|| {
                            anyhow::anyhow!("missing transfer/consistency lower bound")
                        })?;
                        let hi = values.get(&format!("{name}_upper")).ok_or_else(|| {
                            anyhow::anyhow!("missing transfer/consistency upper bound")
                        })?;
                        let point = scalar(value, p)?;
                        let low =
                            Float::with_val_round(p, Float::parse(lo)?, rug::float::Round::Up).0;
                        let high =
                            Float::with_val_round(p, Float::parse(hi)?, rug::float::Round::Down).0;
                        if low > point || high < point {
                            bail!("transfer/consistency point outside enclosure");
                        }
                    }
                    Ok(())
                };
                if id == "finite_section_transfer" {
                    check(&r.values)?;
                    if r.rows.len() != s.modes + 1 {
                        bail!("finite-section prefix count mismatch");
                    }
                    if let Some(c) = input
                        .and_then(|i| i.run_once.as_ref())
                        .and_then(|i| i.comparison.as_ref())
                    {
                        for name in [
                            "comparison_block_frobenius_difference",
                            "comparison_precision_bits",
                        ] {
                            if !r.values.contains_key(name) {
                                bail!("finite-section comparison measurement missing");
                            }
                        }
                        let mut nonzero = false;
                        for value in &c.coefficients {
                            nonzero |= !scalar(value, c.precision_bits)?.is_zero();
                        }
                        if nonzero
                            != r.values
                                .contains_key("comparison_state_signed_block_defect")
                        {
                            bail!("finite-section signed comparison denominator contract mismatch");
                        }
                    }
                }
                for (k, row) in r.rows.iter().enumerate() {
                    if row.values.is_empty() {
                        if id == "finite_section_transfer"
                            || !["missing_input", "cancellation_limited"]
                                .contains(&row.outcome.as_str())
                        {
                            bail!("transfer/consistency measurements missing");
                        }
                        continue;
                    }
                    check(&row.values)?;
                    let core: &[&str] = if id == "finite_section_transfer" {
                        &[
                            "n_modes",
                            "retained_mass",
                            "omitted_mass",
                            "low_residual_squared",
                            "high_forcing_squared",
                            "truncated_energy",
                        ]
                    } else {
                        &["action_difference_norm", "signed_energy_difference"]
                    };
                    for name in core {
                        if !row.values.contains_key(*name) {
                            bail!("transfer/consistency core field missing");
                        }
                    }
                    if id == "finite_section_transfer"
                        && (row.ordinal != k + 1 || scalar(&row.values["n_modes"], p)? != k)
                    {
                        bail!("finite-section prefix order mismatch");
                    }
                }
            }
            if matches!(id, "observable_budget" | "configuration_comparison")
                && !["missing_input", "unresolved"].contains(&r.outcome.as_str())
            {
                let p = options.working_precision_bits;
                let check = |values: &BTreeMap<String, String>| -> Result<()> {
                    let used = scalar(
                        values.get("arithmetic_precision_bits").ok_or_else(|| {
                            anyhow::anyhow!("missing comparison/observation precision")
                        })?,
                        p,
                    )?;
                    if used < p + 64 || used > p + 4096 || !used.is_integer() {
                        bail!("invalid comparison/observation precision");
                    }
                    for (name, value) in values {
                        if name == "arithmetic_precision_bits"
                            || name.ends_with("_lower")
                            || name.ends_with("_upper")
                        {
                            continue;
                        }
                        let lo = values.get(&format!("{name}_lower")).ok_or_else(|| {
                            anyhow::anyhow!("missing comparison/observation lower bound")
                        })?;
                        let hi = values.get(&format!("{name}_upper")).ok_or_else(|| {
                            anyhow::anyhow!("missing comparison/observation upper bound")
                        })?;
                        let point = scalar(value, p)?;
                        let low =
                            Float::with_val_round(p, Float::parse(lo)?, rug::float::Round::Up).0;
                        let high =
                            Float::with_val_round(p, Float::parse(hi)?, rug::float::Round::Down).0;
                        if low > point || high < point {
                            bail!("comparison/observation point outside enclosure");
                        }
                        if id == "observable_budget" {
                            if name.ends_with("_lower_margin") && value != lo {
                                bail!("observation margin is not an outward lower endpoint");
                            }
                            if [
                                "declared_origin_error",
                                "conditional_value_error",
                                "conditional_derivative_error",
                            ]
                            .contains(&name.as_str())
                                && value != hi
                            {
                                bail!("observation error is not an outward upper endpoint");
                            }
                        }
                    }
                    Ok(())
                };
                if id == "observable_budget" {
                    check(&r.values)?;
                    for name in ["transform_origin", "origin_absolute_terms"] {
                        if !r.values.contains_key(name) {
                            bail!("observation origin channel missing");
                        }
                    }
                    if input
                        .and_then(|i| i.run_once.as_ref())
                        .and_then(|i| i.uncertainty.as_ref())
                        .is_some()
                    {
                        for name in [
                            "declared_unit_state_l2_error",
                            "declared_origin_error",
                            "conditional_origin_lower_margin",
                        ] {
                            if !r.values.contains_key(name) {
                                bail!("observation source-error channel missing");
                            }
                        }
                    }
                }
                for row in &r.rows {
                    if row.values.is_empty() {
                        if ![
                            "missing_input",
                            "budget_limited",
                            "cancellation_limited",
                            "unresolved_denominator",
                        ]
                        .contains(&row.outcome.as_str())
                        {
                            bail!("comparison/observation row measurements missing");
                        }
                        continue;
                    }
                    check(&row.values)?;
                    let core: &[&str] = if id == "observable_budget" {
                        &[
                            "t",
                            "value",
                            "derivative",
                            "absolute_value_terms",
                            "absolute_derivative_terms",
                        ]
                    } else {
                        &[
                            "comparison_C",
                            "comparison_N",
                            "comparison_P",
                            "signed_energy_difference",
                        ]
                    };
                    for name in core {
                        if !row.values.contains_key(*name) {
                            bail!("comparison/observation core field missing");
                        }
                    }
                    if id == "observable_budget"
                        && input
                            .and_then(|i| i.run_once.as_ref())
                            .and_then(|i| i.uncertainty.as_ref())
                            .is_some()
                    {
                        for name in [
                            "declared_unit_state_l2_error",
                            "conditional_value_error",
                            "conditional_derivative_error",
                            "conditional_slope_lower_margin",
                        ] {
                            if !row.values.contains_key(name) {
                                bail!("observation root error channel missing");
                            }
                        }
                    }
                }
            }
            if id == "resolution_budget"
                && !["missing_input", "unresolved"].contains(&r.outcome.as_str())
            {
                validate_resolution_report(r, options.working_precision_bits)?;
            }
            if id == "directional_response" {
                for row in &r.rows {
                    if row.values.is_empty() {
                        if ![
                            "missing_input",
                            "budget_limited",
                            "carrier_or_unresolved",
                            "cancellation_limited",
                        ]
                        .contains(&row.outcome.as_str())
                        {
                            bail!("directional measurements missing");
                        }
                        continue;
                    }
                    let p = options.working_precision_bits;
                    for name in [
                        "t",
                        "tau",
                        "directional_energy",
                        "direction_norm_squared",
                        "orthogonality_defect",
                        "rational_root_condition",
                        "root_condition_tolerance",
                        "arithmetic_precision_bits",
                    ] {
                        if !row.values.contains_key(name) {
                            bail!("missing directional field");
                        }
                    }
                    let used = scalar(&row.values["arithmetic_precision_bits"], p)?;
                    if used < p + 64 || used > p + 4096 || !used.is_integer() {
                        bail!("invalid directional arithmetic precision");
                    }
                    for (name, value) in &row.values {
                        if name == "arithmetic_precision_bits"
                            || name.ends_with("_lower")
                            || name.ends_with("_upper")
                        {
                            continue;
                        }
                        let lo = row.values.get(&format!("{name}_lower")).ok_or_else(|| {
                            anyhow::anyhow!("missing directional lower enclosure")
                        })?;
                        let hi = row.values.get(&format!("{name}_upper")).ok_or_else(|| {
                            anyhow::anyhow!("missing directional upper enclosure")
                        })?;
                        let point = scalar(value, p)?;
                        if scalar(lo, p)? > point || scalar(hi, p)? < point {
                            bail!("directional point outside enclosure");
                        }
                    }
                }
            }
            if id == "energy_allowance"
                && [
                    "conditional_bound_expression",
                    "sufficient_bound_unavailable",
                ]
                .contains(&r.outcome.as_str())
            {
                let p = options.working_precision_bits;
                let used = r
                    .values
                    .get("arithmetic_precision_bits")
                    .ok_or_else(|| anyhow::anyhow!("missing allowance arithmetic precision"))?;
                let used = scalar(used, p)?;
                if used < p + 64 || used > p + 4096 || !used.is_integer() {
                    bail!("invalid allowance arithmetic precision");
                }
                for (name, value) in &r.values {
                    if name == "arithmetic_precision_bits"
                        || name.ends_with("_lower")
                        || name.ends_with("_upper")
                    {
                        continue;
                    }
                    let lo = r
                        .values
                        .get(&format!("{name}_lower"))
                        .ok_or_else(|| anyhow::anyhow!("missing allowance lower enclosure"))?;
                    let hi = r
                        .values
                        .get(&format!("{name}_upper"))
                        .ok_or_else(|| anyhow::anyhow!("missing allowance upper enclosure"))?;
                    let point = scalar(value, p)?;
                    // Directed parsing tests the actual decimal endpoint against
                    // the report's stored binary point without nearest-rounding slack.
                    let lo = Float::with_val_round(p, Float::parse(lo)?, rug::float::Round::Up).0;
                    let hi = Float::with_val_round(p, Float::parse(hi)?, rug::float::Round::Down).0;
                    if lo > point || hi < point {
                        bail!("allowance point outside enclosure");
                    }
                }
            }
            if r.outcome != "missing_input" && r.outcome != "unresolved" {
                let expected: Option<Vec<usize>> = match id {
                    "compactness" => Some((1..=options.exponential_rates.len()).collect()),
                    "signed_transform" => {
                        input.map(|i| i.reference_jets.iter().map(|r| r.ordinal).collect())
                    }
                    "directional_response" | "root_transport" | "observable_budget" => {
                        roots.map(|r| r.dataset.points.iter().map(|r| r.ordinal).collect())
                    }
                    "resolution_budget" => {
                        if input.is_some_and(|i| !i.reference_jets.is_empty()) {
                            input.map(|i| i.reference_jets.iter().map(|r| r.ordinal).collect())
                        } else {
                            roots.map(|r| r.dataset.points.iter().map(|r| r.ordinal).collect())
                        }
                    }
                    "arithmetic_energy" => {
                        input.map(|i| (1..=i.arithmetic_component_count()).collect())
                    }
                    "spectral_cluster" => input.map(|i| (1..=i.cluster.len()).collect()),
                    _ => None,
                };
                if expected
                    .as_ref()
                    .is_some_and(|e| !r.rows.iter().map(|r| r.ordinal).eq(e.iter().copied()))
                {
                    bail!("extended report lost or changed an input ordinal");
                }
                let required: &[&str] = match id {
                    "compactness" => &[
                        "transform_origin",
                        "transform_second_derivative",
                        "transform_fourth_derivative",
                    ],
                    "weighted_reference_projection" if r.outcome == "point_measurement" => {
                        &["weighted_l1", "weighted_l2_squared", "signed_integral"]
                    }
                    "signed_transform" => {
                        &["arithmetic_precision_bits", "normalization_precision_bits"]
                    }
                    "arithmetic_energy" => &[
                        "arithmetic_precision_bits",
                        "total_tau_energy_lower",
                        "total_tau_energy_upper",
                        "energy_closure_defect_lower",
                        "energy_closure_defect_upper",
                        "total_tau_energy",
                        "sum_component_energy",
                        "sum_absolute_component_energy",
                        "energy_closure_defect",
                        "operator_action_closure_norm",
                    ],
                    "resolution_budget" => &[
                        "conditional_contiguous_prefix",
                        "relative_tolerance",
                        "finite_curvature_expression",
                    ],
                    "energy_allowance" => &[
                        "upper_trial_energy",
                        "low_block_lower_bound",
                        "high_block_lower_bound",
                        "cross_block_norm_bound",
                        "denominator",
                    ],
                    _ => &[],
                };
                if required.iter().any(|k| !r.values.contains_key(*k)) {
                    bail!("extended report missing required measurements");
                }
                if id == "arithmetic_energy" {
                    let p = options.working_precision_bits;
                    let used = scalar(&r.values["arithmetic_precision_bits"], p)?;
                    if used < p + 64 || used > p + 4096 || !used.is_integer() {
                        bail!("invalid arithmetic energy guard precision");
                    }
                    for values in
                        std::iter::once(&r.values).chain(r.rows.iter().map(|row| &row.values))
                    {
                        for (name, value) in values {
                            if name == "arithmetic_precision_bits"
                                || name.ends_with("_lower")
                                || name.ends_with("_upper")
                            {
                                continue;
                            }
                            let lower = values
                                .get(&format!("{name}_lower"))
                                .ok_or_else(|| anyhow::anyhow!("missing energy lower enclosure"))?;
                            let upper = values
                                .get(&format!("{name}_upper"))
                                .ok_or_else(|| anyhow::anyhow!("missing energy upper enclosure"))?;
                            let point = scalar(value, p)?;
                            if scalar(lower, p)? > point || scalar(upper, p)? < point {
                                bail!("energy point lies outside its reported enclosure");
                            }
                        }
                    }
                }
                if id == "signed_transform"
                    && (scalar(
                        &r.values["arithmetic_precision_bits"],
                        options.working_precision_bits,
                    )? != options.working_precision_bits
                        || scalar(
                            &r.values["normalization_precision_bits"],
                            options.working_precision_bits,
                        )? != options.working_precision_bits + 64)
                {
                    bail!("signed transform arithmetic precision disagrees with input points");
                }
                if id == "energy_allowance" && r.outcome == "conditional_bound_expression" {
                    for name in [
                        "conditional_energy_allowance",
                        "conditional_vector_allowance",
                        "trial_energy_magnitude",
                    ] {
                        if !r.values.contains_key(name) {
                            bail!("energy allowance missing scale interpretation");
                        }
                    }
                    if r.reason.as_ref().is_none_or(|v| v.is_empty()) {
                        bail!("energy allowance requires scale interpretation reason");
                    }
                    let nonzero = scalar(
                        &r.values["trial_energy_magnitude"],
                        options.working_precision_bits,
                    )? > 0;
                    if [
                        "allowance_to_trial_energy_magnitude",
                        "allowance_below_trial_energy_magnitude",
                        "scale_comparison_resolved",
                    ]
                    .iter()
                    .any(|name| r.values.contains_key(*name) != nonzero)
                    {
                        bail!("energy allowance scale comparison disagrees with zero trial energy");
                    }
                }
            }
            xc_core::validate_secret_free(r, "extended research report")?;
            Ok(())
        },
    )?;
    super::research_export::emit(
        &result.value,
        result
            .produced_manifest
            .as_ref()
            .or(result.reused_manifest.as_ref()),
    );
    Ok(result)
}

#[cfg(test)]
mod exhaustive_resumed_contract {
    use super::*;
    #[test]
    fn exhaustive_resumed_signed_channel_rejects_partial_underflow() {
        let p = 128;
        let tiny = Float::with_val(p, 1) << (rug::float::exp_min() - 1);
        assert!(signed_channel_quotient(&tiny, &Float::with_val(p, 1.5), p).is_err());
    }
}

fn validate_resolution_report(r: &ExtendedAnalysis, p: u32) -> Result<()> {
    let check = |values: &BTreeMap<String, String>, skip: &[&str]| -> Result<()> {
        for (name, value) in values {
            if skip.contains(&name.as_str()) || name.ends_with("_lower") || name.ends_with("_upper")
            {
                continue;
            }
            let lo = values
                .get(&format!("{name}_lower"))
                .ok_or_else(|| anyhow::anyhow!("missing resolution lower enclosure"))?;
            let hi = values
                .get(&format!("{name}_upper"))
                .ok_or_else(|| anyhow::anyhow!("missing resolution upper enclosure"))?;
            let point = scalar(value, p)?;
            let lo = Float::with_val_round(p, Float::parse(lo)?, rug::float::Round::Up).0;
            let hi = Float::with_val_round(p, Float::parse(hi)?, rug::float::Round::Down).0;
            if lo > point || hi < point {
                bail!("resolution point outside enclosure");
            }
        }
        Ok(())
    };
    check(&r.values, &["conditional_contiguous_prefix"])?;
    let mut prefix = 0usize;
    let mut contiguous = true;
    for row in &r.rows {
        if row.values.is_empty() {
            if !["missing_input", "cancellation_limited"].contains(&row.outcome.as_str()) {
                bail!("resolution measurements missing");
            }
        } else {
            for name in [
                "t",
                "transform",
                "derivative",
                "absolute_value_terms",
                "absolute_derivative_terms",
                "arithmetic_precision_bits",
            ] {
                if !row.values.contains_key(name) {
                    bail!("missing resolution core field");
                }
            }
            let used = scalar(&row.values["arithmetic_precision_bits"], p)?;
            if used < p + 64 || used > p + 4096 || !used.is_integer() {
                bail!("invalid resolution arithmetic precision");
            }
            check(&row.values, &["arithmetic_precision_bits"])?;
            if row.outcome == "conditional_budget_met" {
                for name in [
                    "conditional_root_distance_allowance",
                    "conditional_slope_margin",
                    "declared_radius",
                    "conditional_budget_target",
                    "declared_source_value_error",
                    "declared_source_derivative_error",
                    "conditional_value_numerator",
                ] {
                    if !row.values.contains_key(name) {
                        bail!("missing conditional resolution field");
                    }
                }
                use super::retained_evidence::finite_math::decimal;
                let bound = decimal(&row.values["conditional_root_distance_allowance"], p + 64)?;
                let target = decimal(&row.values["conditional_budget_target_lower"], p + 64)?;
                if bound.lower() < &0
                    || bound.upper() > target.lower()
                    || scalar(&row.values["conditional_slope_margin_lower"], p)? <= 0
                    || scalar(&row.values["declared_radius_lower"], p)? <= 0
                {
                    bail!("inconsistent conditional resolution qualification");
                }
            }
        }
        if prefix.checked_add(1) != Some(row.ordinal) || row.outcome != "conditional_budget_met" {
            contiguous = false;
        }
        if contiguous {
            prefix = row.ordinal;
        }
    }
    if scalar(
        r.values
            .get("conditional_contiguous_prefix")
            .ok_or_else(|| anyhow::anyhow!("missing conditional resolution prefix"))?,
        p,
    )? != prefix
    {
        bail!("inconsistent conditional resolution prefix");
    }

    Ok(())
}

#[cfg(test)]
mod exhaustive_resumed_resolution_validation_contract {
    use super::*;
    #[test]
    fn exhaustive_resumed_resolution_missing_prefix_returns_error() {
        let report = ExtendedAnalysis {
            diagnostic: "resolution_budget".into(),
            outcome: "point_measurement".into(),
            reason: None,
            lambda_squared: "9".into(),
            n_modes: 0,
            source_precision_bits: 128,
            working_precision_bits: 128,
            convention: "fixture".into(),
            assurance: RESOLUTION_ASSURANCE.into(),
            values: BTreeMap::new(),
            rows: vec![],
        };
        let result = std::panic::catch_unwind(|| validate_resolution_report(&report, 128));
        assert!(
            matches!(result, Ok(Err(_))),
            "missing prefix must fail without panicking"
        );
    }
}
