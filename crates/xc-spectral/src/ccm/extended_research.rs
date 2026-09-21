//! Additional source-bound point diagnostics. External reference values are data,
//! never executable target definitions. No new primary eigensolve is available.
use super::retained_evidence::{
    center, dot, managed, matrix_ancestry, norm2, orientation, precision, scalar, solve_small,
    transform_terms, ResearchRecord, RetainedMatrix, RetainedRoots,
};
use super::state_geometry::RetainedState;
use anyhow::{bail, Result};
use rayon::prelude::*;
use rug::{float::Constant, ops::Pow, Float};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};
use xc_cache::*;
use xc_numerics::prefix::lossless_decimal as dec;

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
            working_precision_bits: s.precision.saturating_add(64),
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
const ASSURANCE:&str="point_diagnostics_only; external_inputs_and_allowances_not_certified; no_RH_or_ground_selection_premise";
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
pub(super) fn source_unit(s: &RetainedState, p: u32) -> Vec<Float> {
    let scale =
        Float::with_val(p, orientation(&s.coefficients, p)) / norm2(&s.coefficients, p).sqrt();
    s.coefficients
        .iter()
        .map(|v| Float::with_val(p, v) * &scale)
        .collect()
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
    let l = scalar(&s.cutoff, p)?.ln();
    let pi = Float::with_val(p, Constant::Pi);
    let v = source_unit(s, p);
    let scale = Float::with_val(p, 1) / l.clone().sqrt();
    let f0 = Float::with_val(p, &v[s.modes]) * &l * &scale;
    let mut m2 = Float::with_val(p, &v[s.modes]) * l.clone().pow(3u32) / 12u32;
    let mut m4 = Float::with_val(p, &v[s.modes]) * l.clone().pow(5u32) / 80u32;
    for (i, a) in v.iter().enumerate() {
        let j = i as i64 - s.modes as i64;
        if j == 0 {
            continue;
        }
        let q = Float::with_val(p, &pi) * j;
        let q2 = q.square();
        m2 += Float::with_val(p, a) * l.clone().pow(3u32) / (Float::with_val(p, &q2) * 2u32);
        m4 += Float::with_val(p, a)
            * l.clone().pow(5u32)
            * (Float::with_val(p, 1) / (Float::with_val(p, &q2) * 4u32)
                - Float::with_val(p, 3) / (q2.square() * 2u32));
    }
    m2 *= &scale;
    m4 *= &scale;
    put(&mut r.values, "transform_origin", &f0);
    put(&mut r.values, "transform_second_derivative", &(-m2.clone()));
    put(&mut r.values, "transform_fourth_derivative", &m4);
    let guard = Float::with_val(p, 1) >> (p - 32);
    if f0.clone().abs() > guard {
        put(
            &mut r.values,
            "sigma",
            &(m2 / (Float::with_val(p, &f0) * 2u32)),
        );
    } else {
        r.outcome = "partial_unresolved".into();
        r.reason = Some("origin denominator unresolved; sigma omitted".into());
    }
    let rates = coeffs(&o.exponential_rates, p)?;
    // Exact finite Fourier integration. Compute the autocorrelation once and
    // reuse it for every weight, avoiding O(grid*N) transcendental quadrature.
    let correlations = (0..v.len())
        .into_par_iter()
        .map(|lag| dot(&v[lag..], &v[..v.len() - lag], p))
        .collect::<Vec<_>>();
    for (i, a) in rates.iter().enumerate() {
        let mut sum = Float::with_val(p, 0);
        let mut absolute = Float::with_val(p, 0);
        if a == &0 {
            sum = correlations[0].clone();
            absolute = sum.clone().abs();
        } else {
            let b = Float::with_val(p, a) * 2u32;
            let endpoint = (Float::with_val(p, a) * &l).exp();
            for (lag, c) in correlations.iter().enumerate() {
                let omega = Float::with_val(p, Constant::Pi) * (2 * lag) / &l;
                let sign = if lag.is_multiple_of(2) { 1 } else { -1 };
                let kernel: Float =
                    Float::with_val(p, &b) * 2u32 * (Float::with_val(p, &endpoint) - sign)
                        / (b.clone().square() + omega.square())
                        / &l;
                let term: Float =
                    Float::with_val(p, c) * kernel * if lag == 0 { 1u32 } else { 2u32 };
                sum += &term;
                absolute += term.abs();
            }
        }
        let mut rr = row(i + 1, "finite_exponential_weighted_norm_squared");
        put(&mut rr.values, "rate", a);
        put(&mut rr.values, "analytic_integral", &sum);
        put(&mut rr.values, "sum_absolute_terms", &absolute);
        rr.notes.push("analytic finite Fourier integral; arithmetic/source error not enclosed; no infinite-support conclusion".into());
        r.rows.push(rr);
    }
    r.convention="unit_L2_dx; signed origin derivatives from analytic Fourier moments; exp(2*a*abs(x)) norm from coefficient autocorrelation; finite support".into();
    Ok(r)
}
fn weighted_projection(
    s: &RetainedState,
    o: &ExtensionOptions,
    i: Option<&ExternalResearchInputs>,
) -> Result<ExtendedAnalysis> {
    let mut r = report("weighted_reference_projection", s, o);
    let Some(t) = i.and_then(|i| i.target.as_ref()) else {
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
    let l = scalar(&s.cutoff, p)?.ln();
    let mut v = s
        .coefficients
        .iter()
        .map(|x| Float::with_val(p, x))
        .collect::<Vec<_>>();
    let c = center(&v, p);
    put(&mut r.values, "source_raw_center", &c);
    put(
        &mut r.values,
        "reference_raw_normalizer",
        &scalar(&t.raw_normalizer, p)?,
    );
    if c.clone().abs() <= Float::with_val(p, 1) >> (p - 32) {
        return Ok(unresolved(r, "source center normalization unresolved"));
    }
    for a in &mut v {
        *a /= &c;
    }
    let target = coeffs(&t.values, p)?;
    if (Float::with_val(p, &target[0]) - 1u32).abs()
        > Float::with_val(p, 1) >> (i.unwrap().precision_bits / 2)
    {
        return Ok(unresolved(
            r,
            "reference values must be normalized to target(1)=1",
        ));
    }
    let basis = t
        .basis_values
        .iter()
        .map(|v| coeffs(v, p))
        .collect::<Result<Vec<_>>>()?;
    let b = basis.len();
    let n = t.intervals;
    let h = Float::with_val(p, &l) / (2 * n);
    let rows = (0..=n)
        .into_par_iter()
        .map(|j| {
            let x = Float::with_val(p, j) * &h;
            let (actual, _) = evaluate(&v, &x, &l, p);
            let difference = actual - &target[j];
            let weight = (Float::with_val(p, &x) / 2u32).exp() * &h
                / if j == 0 || j == n { 2u32 } else { 1u32 };
            (difference, weight)
        })
        .collect::<Vec<_>>();
    let mut gram = vec![Float::with_val(p, 0); b * b];
    let mut rhs = vec![Float::with_val(p, 0); b];
    let mut l1 = Float::with_val(p, 0);
    let mut l2 = Float::with_val(p, 0);
    let mut signed = Float::with_val(p, 0);
    for (j, (d, w)) in rows.iter().enumerate() {
        l1 += d.clone().abs() * w;
        l2 += d.clone().square() * w;
        signed += Float::with_val(p, d) * w;
        for a in 0..b {
            rhs[a] += Float::with_val(p, &basis[a][j]) * d * w;
            for k in 0..b {
                gram[a * b + k] += Float::with_val(p, &basis[a][j]) * &basis[k][j] * w;
            }
        }
    }
    r.convention="f(1)=target(1)=1; x=log(u); 0<=x<=log(C)/2; exp(x/2)dx=du/sqrt(u); nonorthogonal_raw_basis; composite_trapezoid".into();
    put(&mut r.values, "weighted_l1", &l1);
    put(&mut r.values, "weighted_l2_squared", &l2);
    put(&mut r.values, "signed_integral", &signed);
    for a in 0..b {
        put(&mut r.values, &format!("rhs_{a}"), &rhs[a]);
        for k in 0..b {
            put(&mut r.values, &format!("gram_{a}_{k}"), &gram[a * b + k]);
        }
    }
    if b > 0 {
        if let Some((fit, pivot)) = solve_small(&gram, &rhs, p) {
            put(&mut r.values, "minimum_pivot", &pivot);
            let mut residual = Float::with_val(p, 0);
            for (j, (d, w)) in rows.iter().enumerate() {
                let mut rem = d.clone();
                for a in 0..b {
                    rem -= Float::with_val(p, &fit[a]) * &basis[a][j];
                }
                residual += rem.square() * w;
            }
            put(&mut r.values, "fit_residual_norm_squared", &residual);
            for (a, f) in fit.iter().enumerate() {
                put(&mut r.values, &format!("a_{a}"), f);
            }
            if let Some(fixed) = &t.fixed_second_component {
                let fixed = scalar(fixed, p)?;
                put(&mut r.values, "fixed_second_component", &fixed);
                put(
                    &mut r.values,
                    "b2",
                    &(Float::with_val(p, &fit[1]) - &fixed * &fit[0]),
                );
                if fit[0].clone().abs() > Float::with_val(p, 1) >> (p - 32) {
                    put(
                        &mut r.values,
                        "b_effective",
                        &(Float::with_val(p, &fit[1]) / &fit[0]),
                    );
                }
            }
        } else {
            r.outcome = "rank_or_precision_unresolved".into();
            r.reason = Some("nonorthogonal Gram system unresolved".into());
        }
    }
    Ok(r)
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
    let p = o.working_precision_bits;
    let l = scalar(&s.cutoff, p)?.ln();
    let unit = source_unit(s, p);
    let c = center(&unit, p) / l.clone().sqrt();
    if c.clone().abs() <= Float::with_val(p, 1) >> (p - 32) {
        return Ok(unresolved(r, "center normalization unresolved"));
    }
    r.convention="center_one; exp(i*t*x); interior=actual-reference_window; signed_total=interior-exterior_tail; reference_full=window+tail; all channels kept".into();
    r.rows = i
        .reference_jets
        .par_iter()
        .map(|a| -> Result<_> {
            let t = scalar(&a.t, p)?;
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
                let actual = value / &c;
                let window = scalar(w, p)?;
                let full = scalar(f, p)?;
                let tail = scalar(tail, p)?;
                let inside = Float::with_val(p, &actual) - &window;
                let total = Float::with_val(p, &inside) - &tail;
                let closure = Float::with_val(p, &full) - &window - &tail;
                let mut fitted = Float::with_val(p, 0);
                for (k, a) in fit.iter().enumerate() {
                    let v = scalar(a, p)?;
                    put(&mut rr.values, &format!("{name}_fitted_{k}"), &v);
                    fitted += v;
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
                    &(Float::with_val(p, &inside) - fitted),
                );
                let scale = inside.clone().abs() + tail.clone().abs();
                put(
                    &mut rr.values,
                    &format!("{name}_sum_absolute_channels"),
                    &scale,
                );
                if total != 0 && scale > 0 {
                    put(
                        &mut rr.values,
                        &format!("{name}_cancellation_digits"),
                        &(scale / total.clone().abs())
                            .log10()
                            .max(&Float::with_val(p, 0)),
                    );
                }
                if let Some(e) = end {
                    let e = scalar(e, p)?;
                    put(&mut rr.values, &format!("{name}_endpoint_tail_part"), &e);
                    put(
                        &mut rr.values,
                        &format!("{name}_remaining_tail"),
                        &(tail - e),
                    );
                }
                let numerical_floor =
                    (Float::with_val(p, &absolute) / c.clone().abs() + 1u32) >> (p - 32);
                if total.clone().abs() <= numerical_floor {
                    rr.outcome = "cancellation_limited".into();
                }
            }
            rr.notes.push(
                "supplied reference point jets; no infinite-tail or source-error certification"
                    .into(),
            );
            Ok(rr)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(r)
}
struct ParsedOperator {
    diagonal: Vec<Float>,
    dense: Vec<Float>,
    rank: Vec<(Float, Vec<Float>)>,
}
impl ParsedOperator {
    fn from(c: &OperatorComponent, p: u32) -> Result<Self> {
        Ok(Self {
            diagonal: coeffs(&c.diagonal, p)?,
            dense: coeffs(&c.dense, p)?,
            rank: c
                .rank_one
                .iter()
                .map(|r| Ok((scalar(&r.weight, p)?, coeffs(&r.vector, p)?)))
                .collect::<Result<Vec<_>>>()?,
        })
    }
    fn action(&self, v: &[Float], p: u32) -> Vec<Float> {
        let n = v.len();
        let mut out = if self.dense.is_empty() {
            vec![Float::with_val(p, 0); n]
        } else {
            self.dense.par_chunks(n).map(|r| dot(r, v, p)).collect()
        };
        for (a, (d, x)) in out.iter_mut().zip(self.diagonal.iter().zip(v)) {
            *a += Float::with_val(p, d) * x;
        }
        for (weight, q) in &self.rank {
            let factor = dot(q, v, p) * weight;
            for (a, q) in out.iter_mut().zip(q) {
                *a += Float::with_val(p, q) * &factor;
            }
        }
        out
    }
}
pub(super) fn matvec(m: &RetainedMatrix<'_>, v: &[Float], p: u32) -> Vec<Float> {
    m.entries
        .par_chunks(v.len())
        .map(|r| dot(r, v, p))
        .collect()
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
        return Ok(missing(r,"explicit arithmetic component operators required; total Tau does not identify the split"));
    };
    let Some(m) = m else {
        return Ok(missing(r, "retained Tau required for component closure"));
    };
    let p = o.working_precision_bits;
    let v = source_unit(s, p);
    let av = matvec(m, &v, p);
    let total = dot(&v, &av, p);
    let mut combined = vec![Float::with_val(p, 0); v.len()];
    let mut sum = Float::with_val(p, 0);
    let mut absolute = Float::with_val(p, 0);
    let mut actions = i
        .components
        .iter()
        .map(|c| {
            Ok((
                c.label.clone(),
                c.source_digest.clone(),
                ParsedOperator::from(c, p)?.action(&v, p),
                "external operator".to_string(),
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    if i.components.is_empty() {
        if let Some(inputs) = &i.run_once {
            for c in &inputs.component_actions {
                actions.push((
                    c.label.clone(),
                    c.source_digest.clone(),
                    coeffs(&c.action, p)?,
                    c.convention.clone(),
                ));
            }
        }
    }
    for (k, (label, digest, action, convention)) in actions.into_iter().enumerate() {
        let e = dot(&v, &action, p);
        let mut rr = row(k + 1, label);
        put(&mut rr.values, "energy", &e);
        rr.notes
            .push(format!("source {}; {}", digest.0, convention));
        sum += &e;
        absolute += e.abs();
        for (a, b) in combined.iter_mut().zip(action) {
            *a += b;
        }
        r.rows.push(rr);
    }
    let residual = combined
        .iter()
        .zip(&av)
        .map(|(a, b)| (Float::with_val(p, a) - b).square())
        .fold(Float::with_val(p, 0), |a, b| a + b)
        .sqrt();
    put(&mut r.values, "total_tau_energy", &total);
    put(&mut r.values, "sum_component_energy", &sum);
    put(&mut r.values, "sum_absolute_component_energy", &absolute);
    put(
        &mut r.values,
        "energy_closure_defect",
        &(Float::with_val(p, &sum) - &total),
    );
    put(&mut r.values, "operator_action_closure_norm", &residual);
    r.convention = if i.components_are_complete {
        "caller_declares_complete_component_sum; measured_action_closure"
    } else {
        "partial_component_sum; remaining_Tau_action_explicit"
    }
    .into();
    let e = scalar(&s.eigenvalue, p)?;
    put(&mut r.values, "retained_weil_energy", &e);
    let trial = if let Some(c) = i
        .target
        .as_ref()
        .and_then(|t| t.trial_coefficients.as_ref())
    {
        let mut q = coeffs(c, p)?;
        let norm = norm2(&q, p);
        if norm == 0 {
            None
        } else {
            let scale = norm.sqrt();
            for x in &mut q {
                *x /= &scale;
            }
            let energy = dot(&q, &matvec(m, &q, p), p);
            put(&mut r.values, "finite_projected_trial_energy", &energy);
            Some(energy)
        }
    } else {
        None
    };
    if let Some(d) = &i.deficit {
        let d = scalar(d, p)?;
        put(&mut r.values, "reference_deficit", &d);
        if d > 0 {
            put(
                &mut r.values,
                "signed_weil_over_deficit",
                &(Float::with_val(p, &e) / &d),
            );
            if let Some(trial) = &trial {
                put(
                    &mut r.values,
                    "signed_trial_over_deficit",
                    &(Float::with_val(p, trial) / &d),
                );
            }
        } else {
            r.outcome = "partial_unresolved".into();
            r.reason = Some("nonpositive reference deficit; deficit ratios omitted".into());
        }
    }
    if e != 0 {
        if let Some(t) = trial {
            put(&mut r.values, "signed_trial_over_weil", &(t / e));
        }
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
    let v = source_unit(s, p);
    let l = scalar(&s.cutoff, p)?.ln();
    let two_pi = Float::with_val(p, Constant::Pi) * 2u32;
    let e = scalar(&s.eigenvalue, p)?;
    let mut ops = input
        .map(|i| &i.perturbations[..])
        .unwrap_or(&[])
        .iter()
        .map(|c| Ok((&c.label, ParsedOperator::from(c, p)?.action(&v, p))))
        .collect::<Result<Vec<_>>>()?;
    if let Some(inputs) = input.and_then(|i| i.run_once.as_ref()) {
        for a in &inputs.derivative_actions {
            if ops.iter().any(|(label, _)| *label == &a.label) {
                bail!("duplicate derivative action label");
            }
            ops.push((&a.label, coeffs(&a.action, p)?));
        }
    }
    // Explicit arithmetic work budget. Every omitted row remains in the report.
    let limit = o.maximum_directional_rows;
    r.convention="even finite state; tau=(t*log(C)/(2*pi))^2; x=(tau-j^2)^-1*v-v*(v^T(tau-j^2)^-1*v); kappa=x^T(Tau-EI)x; response ratios conditional on simple-minimum/displacement/root hypotheses".into();
    let checkpoints = super::capture_runtime::Checkpoints::new(&(
        "directional-rows-v2",
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
            let t = scalar(t, p)?;
            let tau = (Float::with_val(p, &t) * &l / &two_pi).square();
            put(&mut rr.values, "t", &t);
            put(&mut rr.values, "tau", &tau);
            let floor = (tau.clone().abs() + 1u32) >> (p - 32);
            let mut rv = Vec::with_capacity(v.len());
            for (idx, a) in v.iter().enumerate() {
                let j = idx.abs_diff(s.modes);
                let denominator = Float::with_val(p, &tau) - j * j;
                if denominator.clone().abs() <= floor {
                    rr.outcome = "carrier_or_unresolved".into();
                    return Ok(rr);
                }
                rv.push(Float::with_val(p, a) / denominator);
            }
            let root_condition = rv.iter().fold(Float::with_val(p, 0), |mut a, b| {
                a += b;
                a
            });
            let root_scale = rv
                .iter()
                .fold(Float::with_val(p, 0), |a, b| a + b.clone().abs());
            let root_resolved = root_condition.clone().abs()
                <= (root_scale + 1u32) >> (s.precision.saturating_sub(32));
            let vr = dot(&v, &rv, p);
            let x = rv
                .iter()
                .zip(&v)
                .map(|(a, b)| Float::with_val(p, a) - Float::with_val(p, &vr) * b)
                .collect::<Vec<_>>();
            let ax = matvec(m, &x, p);
            let x2 = norm2(&x, p);
            let k = dot(&x, &ax, p) - Float::with_val(p, &e) * &x2;
            put(&mut rr.values, "directional_energy", &k);
            put(&mut rr.values, "direction_norm_squared", &x2);
            put(&mut rr.values, "orthogonality_defect", &dot(&x, &v, p));
            put(&mut rr.values, "rational_root_condition", &root_condition);
            let guard = (dot(
                &x.iter().map(|v| v.clone().abs()).collect::<Vec<_>>(),
                &ax.iter().map(|v| v.clone().abs()).collect::<Vec<_>>(),
                p,
            ) + Float::with_val(p, &e).abs() * &x2
                + 1u32)
                >> (p - 32);
            if k.clone().abs() <= guard {
                rr.outcome = "unresolved_denominator".into();
            }
            for (label, action) in &ops {
                let forcing = dot(&x, action, p);
                put(&mut rr.values, &format!("forcing_{label}"), &forcing);
                if k.clone().abs() > guard && root_resolved {
                    put(
                        &mut rr.values,
                        &format!("conditional_tau_response_{label}"),
                        &(-forcing / &k),
                    );
                }
            }
            if !root_resolved {
                if rr.outcome == "point_measurement" {
                    rr.outcome = "channels_resolved_budget_unassessed".into();
                }
                rr.notes.push("point does not resolve the unshifted rational root condition; response ratios withheld".into());
            }
            if ops.is_empty() {
                if rr.outcome == "point_measurement" {
                    rr.outcome = "channels_resolved_budget_unassessed".into();
                }
                rr.notes
                    .push("perturbation operators unavailable; forcing ratios not computed".into());
            }
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
        "{}; caller-declared partitions; finite supplied atoms only; {}",
        i.atom_coordinate.as_deref().unwrap(),
        i.atom_coverage.as_deref().unwrap()
    );
    let mut checkpoints = coeffs(&i.tail_checkpoints, p)?;
    if checkpoints.is_empty() {
        for a in &i.atoms {
            checkpoints.push(scalar(&a.coordinate, p)?);
        }
        checkpoints.sort_by(|a, b| a.total_cmp(b));
        checkpoints.dedup();
        let all = checkpoints;
        checkpoints = all
            .iter()
            .enumerate()
            .filter(|(k, _)| (*k + 1).is_power_of_two() || *k + 1 == all.len())
            .map(|(_, v)| v.clone())
            .collect();
    }
    checkpoints.sort_by(|a, b| a.total_cmp(b));
    checkpoints.dedup();
    if checkpoints.iter().any(|x| x < &0) {
        bail!("tail checkpoints must be nonnegative");
    }
    let mut families: BTreeMap<(String, String), Vec<(Float, Float)>> = BTreeMap::new();
    for a in &i.atoms {
        families
            .entry((a.family.clone(), a.partition.clone()))
            .or_default()
            .push((scalar(&a.coordinate, p)?, scalar(&a.weight, p)?));
    }
    for ((family, partition), mut atoms) in families {
        atoms.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut pos = 0;
        let mut mass = Float::with_val(p, 0);
        let mut abs = Float::with_val(p, 0);
        let mut moments = vec![Float::with_val(p, 0); 3];
        let mut inverse_defined = true;
        let total = atoms.iter().fold(Float::with_val(p, 0), |mut s, (_, w)| {
            s += w;
            s
        });
        for cutoff in &checkpoints {
            while pos < atoms.len() && &atoms[pos].0 <= cutoff {
                let (x, w) = &atoms[pos];
                mass += w;
                abs += w.clone().abs();
                if x == &0 {
                    inverse_defined &= w == &0;
                } else {
                    let inverse = Float::with_val(p, 1) / x;
                    let mut term = Float::with_val(p, w);
                    for m in &mut moments {
                        term *= &inverse;
                        *m += &term;
                    }
                }
                pos += 1;
            }
            let mut rr = row(r.rows.len() + 1, format!("{family}/{partition}"));
            put(&mut rr.values, "cutoff", cutoff);
            put(&mut rr.values, "included_mass", &mass);
            put(&mut rr.values, "included_absolute_mass", &abs);
            put(
                &mut rr.values,
                "remaining_supplied_mass",
                &(Float::with_val(p, &total) - &mass),
            );
            put(&mut rr.values, "included_count", &Float::with_val(p, pos));
            for (k, m) in moments.iter().enumerate() {
                if inverse_defined {
                    put(
                        &mut rr.values,
                        &format!("weighted_inverse_moment_{}", k + 1),
                        m,
                    );
                }
            }
            if !inverse_defined {
                rr.outcome = "unresolved_denominator".into();
                rr.notes.push("inverse moments undefined: the included lattice origin has nonzero mass; origin retained in mass and count".into());
            }
            rr.notes.push(
                "unprovided infinite tail not bounded; known-zero geometry remains an input".into(),
            );
            r.rows.push(rr);
        }
    }
    Ok(r)
}
fn padded_unit(v: &[Float], modes: usize, p: u32) -> Vec<Float> {
    let mut result = vec![Float::with_val(p, 0); 2 * modes + 1];
    let offset = modes - v.len() / 2;
    let n = norm2(v, p).sqrt();
    for (j, v) in v.iter().enumerate() {
        result[offset + j] = Float::with_val(p, v) / &n;
    }
    result
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
    let v = padded_unit(&s.coefficients, modes, p);
    let q = i
        .cluster
        .iter()
        .map(|c| Ok(padded_unit(&coeffs(&c.coefficients, p)?, modes, p)))
        .collect::<Result<Vec<_>>>()?;
    let b = q.len();
    let mut gram = Vec::with_capacity(b * b);
    let mut rhs = Vec::with_capacity(b);
    for a in &q {
        rhs.push(dot(a, &v, p));
        for b in &q {
            gram.push(dot(a, b, p));
        }
    }
    for a in 0..b {
        put(&mut r.values, &format!("source_overlap_{a}"), &rhs[a]);
        for k in 0..b {
            put(&mut r.values, &format!("gram_{a}_{k}"), &gram[a * b + k]);
        }
    }
    if let Some((coeff, pivot)) = solve_small(&gram, &rhs, p) {
        let mut residual = v.clone();
        for (a, c) in q.iter().zip(&coeff) {
            for (x, y) in residual.iter_mut().zip(a) {
                *x -= Float::with_val(p, c) * y;
            }
        }
        put(
            &mut r.values,
            "source_cluster_leakage_squared",
            &norm2(&residual, p),
        );
        put(&mut r.values, "minimum_gram_pivot", &pivot);
    } else {
        r.outcome = "rank_or_precision_unresolved".into();
        r.reason = Some("cluster Gram system unresolved".into());
    }
    let old = i
        .previous_cluster
        .iter()
        .map(|c| Ok(padded_unit(&coeffs(&c.coefficients, p)?, modes, p)))
        .collect::<Result<Vec<_>>>()?;
    for (a, v) in q.iter().enumerate() {
        let mut rr = row(a + 1, "current_cluster_vector");
        put(
            &mut rr.values,
            "eigenvalue",
            &scalar(&i.cluster[a].eigenvalue, p)?,
        );
        let mut matches = Vec::new();
        for (b, u) in old.iter().enumerate() {
            let overlap = dot(v, u, p);
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
                put(
                    &mut rr.values,
                    "match_margin",
                    &(Float::with_val(p, overlap) - &matches[1].1),
                );
            }
        }
        rr.notes.push("cross-N zero-padding in the same cutoff Fourier basis; overlap matching does not certify branch identity".into());
        r.rows.push(rr);
    }
    if let Some([low, high]) = &i.cluster_boundary_eigenvalues {
        let low = scalar(low, p)?;
        let high = scalar(high, p)?;
        put(&mut r.values, "declared_boundary_gap", &(high - low));
    }
    r.convention="unit coefficient norm; common-cutoff Fourier embedding; nonorthogonal cluster Gram projection; externally retained vector points".into();
    Ok(r)
}
fn resolution(
    s: &RetainedState,
    roots: Option<&RetainedRoots>,
    o: &ExtensionOptions,
    input: Option<&ExternalResearchInputs>,
) -> Result<ExtendedAnalysis> {
    let mut r = report("resolution_budget", s, o);
    let p = o.working_precision_bits;
    let reference = input.map(|i| i.reference_jets.as_slice()).unwrap_or(&[]);
    if reference.is_empty() && roots.is_none() {
        return Ok(missing(r, "retained roots or reference ordinates required"));
    }
    let points: Vec<(usize, Option<String>, String)> = if reference.is_empty() {
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
    let l = scalar(&s.cutoff, p)?.ln();
    let curvature = (l.clone().pow(5u32) / 80u32).sqrt();
    let tol = scalar(&o.relative_tolerance, p)?;
    let mut qualifying = 0usize;
    let mut contiguous = true;
    let mut previous = None;
    for (ordinal, t, status) in points {
        let mut rr = row(ordinal, status.clone());
        let Some(t) = t else {
            rr.outcome = "missing_input".into();
            contiguous = false;
            r.rows.push(rr);
            continue;
        };
        let t = scalar(&t, p)?;
        let neighbors = reference
            .iter()
            .filter(|j| j.ordinal.abs_diff(ordinal) == 1)
            .map(|j| Ok((j.ordinal, scalar(&j.t, p)?)))
            .collect::<Result<Vec<_>>>()?;
        let ordered = neighbors.iter().all(|(index, value)| {
            if *index < ordinal {
                value < &t
            } else {
                value > &t
            }
        });
        let spacing = if ordered {
            neighbors
                .iter()
                .map(|(_, v)| (Float::with_val(p, v) - &t).abs())
                .min_by(Float::total_cmp)
        } else {
            rr.notes.push(
                "reference neighbors are not strictly ordered; spacing ratios withheld".into(),
            );
            None
        };
        put(
            &mut rr.values,
            "supplied_adjacent_reference_count",
            &Float::with_val(p, neighbors.len()),
        );
        if let Some(gap) = &spacing {
            put(&mut rr.values, "reference_neighbor_spacing", gap);
            rr.notes.push("spacing uses supplied adjacent reference ordinals; no zeta identification is inferred".into());
        }
        if let Some(j) = reference.iter().find(|j| j.ordinal == ordinal) {
            if let Some(join) = j.matched_root_ordinal {
                if let Some(point) = roots
                    .and_then(|r| r.dataset.points.iter().find(|v| v.ordinal == join))
                    .and_then(|v| v.value.as_ref())
                {
                    let delta = scalar(point, p)? - &t;
                    put(
                        &mut rr.values,
                        "matched_retained_root_ordinal",
                        &Float::with_val(p, join),
                    );
                    put(
                        &mut rr.values,
                        "matched_root_reference_displacement",
                        &delta,
                    );
                    if let Some(gap) = &spacing {
                        put(
                            &mut rr.values,
                            "spacing_normalized_displacement",
                            &(delta / gap),
                        );
                    }
                    rr.notes.push("root join declared by caller and bound to retained source; not certified zero identification".into());
                } else {
                    rr.notes.push(
                        "declared retained-root join unavailable; displacement unassessed".into(),
                    );
                }
            }
        }
        let (f, d, a, ad) = transform_terms(s, &t, p)?;
        put(&mut rr.values, "t", &t);
        put(&mut rr.values, "transform", &f);
        put(&mut rr.values, "derivative", &d);
        put(&mut rr.values, "absolute_value_terms", &a);
        put(&mut rr.values, "absolute_derivative_terms", &ad);
        let gf = (Float::with_val(p, &a) + 1u32) >> (p - 32);
        let gd = (Float::with_val(p, &ad) + 1u32) >> (p - 32);
        if d.clone().abs() <= gd {
            rr.outcome = "unresolved_derivative".into();
        } else {
            put(
                &mut rr.values,
                "point_newton_correction",
                &(-Float::with_val(p, &f) / &d),
            );
            if let Some(gap) = &spacing {
                put(
                    &mut rr.values,
                    "spacing_normalized_newton_correction",
                    &(-Float::with_val(p, &f) / &d / gap),
                );
            }
            rr.outcome = if f.clone().abs() <= gf {
                "cancellation_limited"
            } else {
                "channels_resolved_budget_unassessed"
            }
            .into();
            if let Some(j) = reference.iter().find(|j| j.ordinal == ordinal) {
                // Error allowances refer to unit-L2 actual transform channels here.
                if let (Some(value_error), Some(derivative_error), Some(radius)) = (
                    &j.source_value_error,
                    &j.source_derivative_error,
                    &j.root_separation_radius,
                ) {
                    let ev = scalar(value_error, p)?;
                    let ed = scalar(derivative_error, p)?;
                    let h = scalar(radius, p)?;
                    let slope = d.clone().abs() - &ed - &gd - &curvature * &h;
                    put(&mut rr.values, "declared_source_value_error", &ev);
                    put(&mut rr.values, "declared_source_derivative_error", &ed);
                    if let Some(tail) = &j.tail_value_error {
                        put(
                            &mut rr.values,
                            "declared_reference_tail_error",
                            &scalar(tail, p)?,
                        );
                    }
                    let numerator = f.clone().abs() + ev + &gf;
                    put(&mut rr.values, "declared_radius", &h);
                    put(&mut rr.values, "conditional_slope_margin", &slope);
                    if h > 0 && slope > 0 {
                        let bound = numerator / &slope;
                        put(
                            &mut rr.values,
                            "conditional_root_distance_allowance",
                            &bound,
                        );
                        if let Some(gap) = &spacing {
                            put(
                                &mut rr.values,
                                "spacing_normalized_conditional_allowance",
                                &(Float::with_val(p, &bound) / gap),
                            );
                        }
                        rr.outcome = if bound <= Float::with_val(p, &h) * &tol {
                            "conditional_budget_met"
                        } else {
                            "conditional_budget_not_met"
                        }
                        .into();
                    } else {
                        rr.outcome = "conditional_budget_unresolved".into();
                    }
                    rr.notes.push("conditional on supplied absolute source errors, valid curvature and isolation interval; supplied bounds not certified here".into());
                }
            }
        }
        if previous.is_none() && ordinal != 1 || previous.is_some_and(|x| ordinal != x + 1) {
            contiguous = false;
        }
        if rr.outcome != "conditional_budget_met" {
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
    put(&mut r.values, "relative_tolerance", &tol);
    put(&mut r.values, "finite_curvature_expression", &curvature);
    r.convention="unit_L2_dx actual transform; retained-root ordinals are not zeta ordinals; conditional finite-source radius budget; not minimum-N law".into();
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
    let u = scalar(&a.upper_trial_energy, p)?;
    let b = scalar(&a.low_block_lower_bound, p)?;
    let mu = scalar(&a.high_block_lower_bound, p)?;
    let h = scalar(&a.cross_block_norm_bound, p)?;
    let d = Float::with_val(p, &mu) - &u;
    for (name, v) in [
        ("upper_trial_energy", &u),
        ("low_block_lower_bound", &b),
        ("high_block_lower_bound", &mu),
        ("cross_block_norm_bound", &h),
        ("denominator", &d),
    ] {
        put(&mut r.values, name, v);
    }
    if d > 0 {
        let energy_allowance = h.clone().square() / &d;
        put(
            &mut r.values,
            "conditional_energy_allowance",
            &energy_allowance,
        );
        put(&mut r.values, "conditional_vector_allowance", &(h / d));
        r.outcome = "conditional_bound_expression".into();
        // Compare scales without implying a relative error bound for an
        // eigenvalue. U is a declared trial upper energy, not ground truth.
        let magnitude = u.abs();
        put(&mut r.values, "trial_energy_magnitude", &magnitude);
        if magnitude > 0 {
            put(
                &mut r.values,
                "allowance_to_trial_energy_magnitude",
                &(Float::with_val(p, &energy_allowance) / &magnitude),
            );
            let below = energy_allowance < magnitude;
            put(
                &mut r.values,
                "allowance_below_trial_energy_magnitude",
                &Float::with_val(p, u32::from(below)),
            );
            r.reason = Some(if below {
                "allowance is smaller than the trial energy magnitude; scale comparison only, not a relative eigenvalue error certificate"
            } else {
                "non-informative at the trial energy scale: allowance is at least the trial energy magnitude; not evidence of relative energy accuracy"
            }.into());
        } else {
            r.reason = Some("trial energy is zero; relative scale comparison unavailable; no division by the trial energy performed".into());
        }
    } else {
        r.outcome = "sufficient_bound_unavailable".into();
        r.reason = Some(
            "high block lower bound is not above trial upper energy; no division performed".into(),
        );
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
        "arithmetic_energy" => input.map_or(0, |i| {
            i.components
                .len()
                .max(i.run_once.as_ref().map_or(0, |r| r.component_actions.len()))
        }),
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
            * (8192 + 64 * (u64::from(options.working_precision_bits) / 3 + 32))
            > options.maximum_estimated_output_bytes;

    let result = managed(
        kind,
        {
            let mut request = json!({"semantics":if id == "transform_enclosure" { "extended-retained-diagnostics-v3" } else if matches!(id,"band_reconstruction"|"tail_operator"|"weighted_tail") { "extended-retained-diagnostics-v4" } else if id == "resolution_budget" { "extended-retained-diagnostics-v3" } else { "extended-retained-diagnostics-v2" },"expected_rows":if matches!(id,"transform_enclosure"|"operator_cluster"|"finite_section_transfer"|"configuration_comparison"|"weighted_tail"|"band_reconstruction"|"tail_operator") { None } else { Some(estimated_rows) },"diagnostic":id,"options":options,"external_input_digest":input_digest,"state_selection_policy":s.selection_policy,"state_manifest_tags":s.manifest.tags});
            if id == "band_reconstruction" {
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
                request["enclosure_algorithm"] = json!("centered-taylor-48-integral-remainder-v2");
            }
            if id == "energy_allowance" {
                request["allowance_interpretation"] = json!("trial-energy-scale-v1");
            }
            request
        },
        &sources,
        cache,
        || {
            let _stage = super::capture_runtime::Stage::new(format!("{id} compute"));
            if budget_exceeded {
                return Ok(unresolved(report(id,s,options),"row/output resource budget exceeded; retry this diagnostic with explicit limits"));
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
                    "arithmetic_energy" => input.map(|i| {
                        (1..=i
                            .components
                            .len()
                            .max(i.run_once.as_ref().map_or(0, |r| r.component_actions.len())))
                            .collect()
                    }),
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
                    "arithmetic_energy" => &[
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
