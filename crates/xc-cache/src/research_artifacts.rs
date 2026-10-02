//! Managed research records. Full plans and hypothesis packets are private-only.
//! Measurement bytes are retained even when another requested diagnostic fails.
use crate::*;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use xc_core::{CaptureReceipt, DiagnosticOutcome, EvidenceRef};

pub const CAPTURE_RECEIPT_KIND: &str = "research_capture_receipt";
pub const HYPOTHESIS_EVALUATION_KIND: &str = "research_hypothesis_evaluation";
pub const CAPTURE_RECORD_SEMANTICS: &str = "research-capture-record-v1";

fn invalid(message: impl Into<String>) -> CacheError {
    CacheError::InvalidManifest(message.into())
}

/// A terminal acquisition outcome. Numerical nonacceptance belongs in the
/// measurement payload; it does not mean that acquisition failed.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "status", deny_unknown_fields)]
pub enum CaptureFailure {
    Missing { reason: String },
    Blocked { reason: String },
    Failed { reason: String },
}
impl CaptureFailure {
    /// Preserve actionable errors in private receipts unless they contain secrets.
    pub fn failed(error: impl std::fmt::Display) -> Self {
        Self::Failed {
            reason: safe_failure_reason(error),
        }
    }

    fn outcome(self) -> DiagnosticOutcome {
        match self {
            Self::Missing { reason } => DiagnosticOutcome::Missing { reason },
            Self::Blocked { reason } => DiagnosticOutcome::Blocked { reason },
            Self::Failed { reason } => DiagnosticOutcome::Failed { reason },
        }
    }
}

fn safe_failure_reason(error: impl std::fmt::Display) -> String {
    let reason = error.to_string();
    if reason.trim().is_empty()
        || xc_core::validate_secret_free(&reason, "capture failure").is_err()
    {
        "diagnostic returned an empty or secret-bearing error; details omitted".into()
    } else {
        reason
    }
}

/// Numerical coverage is separate from successful acquisition. Unknown legacy
/// payloads remain unassessed; a saved conditional expression is not a resolved result.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericalCoverage {
    pub outcome: String,
    pub retained_rows: usize,
    pub expected_rows: Option<usize>,
    pub resolved_rows: usize,
    pub qualified_rows: usize,
    pub unresolved_rows: usize,
    pub row_outcomes: BTreeMap<String, usize>,
    pub reason: Option<String>,
    pub recovery: Option<String>,
}
impl NumericalCoverage {
    pub fn from_value(value: &serde_json::Value) -> Self {
        let data = value.get("data").unwrap_or(value);
        let mut result = Self {
            outcome: "unassessed".into(),
            reason: data["reason"].as_str().map(str::to_owned),
            expected_rows: value["request"]["expected_rows"]
                .as_u64()
                .and_then(|n| n.try_into().ok()),
            ..Self::default()
        };
        if let Some(rows) = data["rows"].as_array() {
            result.retained_rows = rows.len();
            for row in rows {
                let outcome = row["outcome"].as_str().unwrap_or("unassessed");
                *result.row_outcomes.entry(outcome.into()).or_default() += 1;
                match outcome {
                    "point_measurement" | "certified_finite_enclosure" => result.resolved_rows += 1,
                    "conditional_budget_met"
                    | "conditional_budget_not_met"
                    | "channels_resolved_budget_unassessed"
                    | "computed_not_certified" => result.qualified_rows += 1,
                    _ => result.unresolved_rows += 1,
                }
            }
        }
        let mut default_outcome =
            if result.retained_rows > 0 && result.resolved_rows == result.retained_rows {
                "point_measurement"
            } else {
                "unassessed"
            };
        let labelled = matches!(
            data["outcome"].as_str(),
            Some("computed" | "computed_with_limitation")
        );
        let mut outcome_override = None;
        if data["rows"].as_array().is_none_or(|rows| rows.is_empty()) {
            if let Some(shape) = shape_rows(data) {
                result.retained_rows = shape.retained;
                result.resolved_rows = shape.resolved;
                result.qualified_rows = shape.qualified;
                result.unresolved_rows = shape.retained - shape.resolved - shape.qualified;
                for (label, count) in [
                    (shape.outcome, shape.resolved),
                    ("qualified", shape.qualified),
                    ("unresolved", result.unresolved_rows),
                ] {
                    if count > 0 {
                        result.row_outcomes.insert(label.into(), count);
                    }
                }
                if shape.retained > 0 {
                    default_outcome = shape.outcome;
                }
                if labelled || data["outcome"].is_null() {
                    outcome_override = Some(default_outcome);
                }
            }
        }
        let outcome = outcome_override
            .or(data["outcome"].as_str())
            .unwrap_or(default_outcome);
        result.outcome = if result.unresolved_rows > 0
            || result
                .expected_rows
                .is_some_and(|n| n > result.retained_rows)
        {
            "partial_unresolved"
        } else if result.qualified_rows > 0 {
            "qualified"
        } else {
            outcome
        }
        .into();
        if result.retained_rows == 0
            && matches!(
                result.outcome.as_str(),
                "point_measurement" | "certified_finite_enclosure"
            )
        {
            result.outcome = "unassessed".into();
            result.reason = Some("a result was reported without any countable rows".into());
        }
        if !matches!(
            result.outcome.as_str(),
            "point_measurement" | "certified_finite_enclosure"
        ) {
            result.recovery = Some("inspect retained outcome and prerequisite fields; rerun only this diagnostic with corrected inputs or an explicit resource/precision policy; reuse primary sources".into());
        }
        result
    }
}

/// Coverage rows of a measurement payload that carries no generic
/// `outcome`/`rows` fields, recognized by its documented shape.
struct ShapeRows {
    retained: usize,
    resolved: usize,
    qualified: usize,
    /// Outcome of a resolved row.
    outcome: &'static str,
}

fn shape(
    retained: usize,
    resolved: usize,
    qualified: usize,
    outcome: &'static str,
) -> Option<ShapeRows> {
    Some(ShapeRows {
        retained,
        resolved,
        qualified,
        outcome,
    })
}

fn strings(data: &serde_json::Value, keys: &[&str]) -> bool {
    keys.iter().all(|key| data[*key].is_string())
}

fn count(items: &[serde_json::Value], resolved: impl Fn(&serde_json::Value) -> bool) -> usize {
    items.iter().filter(|item| resolved(item)).count()
}

/// Rows per payload shape:
/// - eigenfunction profile: samples; target distance and residual analysis:
///   quadrature rules; resolution evidence: refinements plus its reported and
///   ladder verdicts; deviation decomposition: metric projections;
/// - evenness, operator energy, state geometry, retained reduction and a
///   prefix checkpoint: one row; sector gap analysis: one row; sector
///   spectrum: one row per retained eigenpair;
/// - prefix ladder: checkpoints plus the ladder's completion;
/// - root band: retained roots plus missing roots; root conditioning: roots;
///   prime-power and u-flow responses: roots plus events or channels;
/// - sector-gap certificate: one certified row when it certifies a simple
///   finite ground state and a positive definite finite matrix;
/// - target comparison: grid levels, qualified when a limitation is recorded.
///
/// A prefix checkpoint that exports only the innovation (the predeclared
/// eigenstate exclusion) is qualified. `None` for other shapes.
fn shape_rows(data: &serde_json::Value) -> Option<ShapeRows> {
    const CONVERGED: &str = "converged";
    const EXPORTED: &str = "export_checks_passed";
    const INNOVATION_ONLY: &str = "innovation_export_passed_eigenpair_not_supplied";
    let status = |row: &serde_json::Value, value: &str| row["status"].as_str() == Some(value);
    if let Some(items) = data.as_array() {
        let mut total = ShapeRows {
            retained: 0,
            resolved: 0,
            qualified: 0,
            outcome: "point_measurement",
        };
        for (index, item) in items.iter().enumerate() {
            let rows = shape_rows(item)?;
            if index == 0 {
                total.outcome = rows.outcome;
            } else if total.outcome != rows.outcome {
                return None;
            }
            total.retained += rows.retained;
            total.resolved += rows.resolved;
            total.qualified += rows.qualified;
        }
        return Some(total);
    }
    if let (Some(bounds), true) = (
        data["exact_form_bounds"].as_array(),
        data.get("full").is_some(),
    ) {
        let sectors = ["full", "even_sector", "odd_sector"];
        let resolved = count(bounds, |row| {
            strings(row, &["exact_form_lower", "exact_form_upper"])
        }) + sectors
            .iter()
            .filter(|key| data[**key]["spectral_upper"].is_string())
            .count();
        return shape(
            bounds.len() + sectors.len(),
            resolved,
            0,
            "certified_finite_enclosure",
        );
    }
    if data.get("signed_unit_overlap").is_some() {
        // The overlap with the reference, plus one row per fit coefficient
        // when a fit basis was supplied; a requested fit that did not solve is
        // unresolved.
        let basis = data["rhs"].as_array().map_or(0, Vec::len);
        let overlap = usize::from(strings(
            data,
            &["signed_unit_overlap", "difference_norm_squared"],
        ));
        let fitted = data["coefficients"]
            .as_array()
            .map_or(0, |values| count(values, |value| value.is_string()));
        return shape(1 + basis, overlap + fitted, 0, "point_measurement");
    }
    if let Some(evidence) = data.get("retained_evidence") {
        let rows = shape_rows(evidence)?;
        let verdicts = [
            "reported_resolution_tolerance_met",
            "refinement_ladder_tolerance_met",
        ];
        let met = verdicts
            .iter()
            .filter(|key| data[**key].as_bool() == Some(true))
            .count();
        return shape(
            rows.retained + verdicts.len(),
            rows.resolved + met,
            rows.qualified,
            rows.outcome,
        );
    }
    if let Some(rows) = data["refinements"].as_array() {
        let resolved = count(rows, |row| row["tolerance_met"].as_bool() == Some(true));
        return shape(rows.len(), resolved, 0, "point_measurement");
    }
    if let Some(rows) = data["projections"].as_array() {
        let resolved = count(rows, |row| {
            strings(row, &["amplitude", "relative_residual"])
        });
        return shape(rows.len(), resolved, 0, "point_measurement");
    }
    if let Some(levels) = data["levels"].as_array() {
        return match data["outcome"].as_str() {
            Some("computed") => shape(levels.len(), levels.len(), 0, "point_measurement"),
            Some("computed_with_limitation") => {
                shape(levels.len(), 0, levels.len(), "point_measurement")
            }
            _ => None,
        };
    }
    if let Some(rows) = data["measurements"].as_array() {
        let keys = ["distance_to_target", "absolute_residual_mass"];
        if !rows
            .iter()
            .all(|row| keys.iter().any(|key| row.get(*key).is_some()))
        {
            return None;
        }
        let resolved = count(rows, |row| keys.iter().any(|key| row[*key].is_string()));
        return shape(rows.len(), resolved, 0, "point_measurement");
    }
    if let (Some(u), Some(f)) = (data["u_values"].as_array(), data["f_values"].as_array()) {
        let resolved = if u.len() == f.len() {
            count(f, |value| value.is_string())
        } else {
            0
        };
        return shape(f.len(), resolved, 0, "point_measurement");
    }
    if data.get("evenness_deviation").is_some() {
        let ok = strings(
            data,
            &[
                "evenness_deviation",
                "natural_eigenvalue",
                "forced_eigenvalue",
            ],
        );
        return shape(1, usize::from(ok), 0, "point_measurement");
    }
    if data.get("gap_log").is_some() && data.get("even_simple").is_some() {
        let ok = strings(data, &["gap_log", "lambda_even", "lambda_odd"])
            && data["even_simple"].is_boolean();
        return shape(1, usize::from(ok), 0, "point_measurement");
    }
    if let (Some(rows), Some(ladder)) = (data["checkpoints"].as_array(), data.get("ladder")) {
        let complete = usize::from(ladder.get("stopped").is_some_and(|v| v.is_null()));
        return shape(
            rows.len() + 1,
            count(rows, |row| status(row, EXPORTED)) + complete,
            count(rows, |row| status(row, INNOVATION_ONLY)),
            "point_measurement",
        );
    }
    if data.get("squared_overlap").is_some()
        && data.get("dimension").is_some()
        && data.get("status").is_some()
    {
        return shape(
            1,
            usize::from(status(data, EXPORTED)),
            usize::from(status(data, INNOVATION_ONLY)),
            "point_measurement",
        );
    }
    if let (Some(passed), true) = (
        data["checks_passed"].as_bool(),
        data.get("computed_eigenvalues").is_some(),
    ) {
        return shape(1, usize::from(passed), 0, "point_measurement");
    }
    if let (Some(points), Some(missing)) =
        (data["points"].as_array(), data["missing_count"].as_u64())
    {
        let missing = usize::try_from(missing).ok()?;
        let resolved = count(points, |row| {
            row["source_status"].as_str() == Some(CONVERGED)
        });
        return shape(points.len() + missing, resolved, 0, "point_measurement");
    }
    if let (Some(rows), true) = (
        data["outcomes"].as_array(),
        data.get("root_count").is_some(),
    ) {
        return shape(
            rows.len(),
            count(rows, |row| status(row, CONVERGED)),
            0,
            "point_measurement",
        );
    }
    if let Some(roots) = data["roots"].as_array() {
        if let Some(extra) = data["events"].as_array().or(data["channels"].as_array()) {
            let resolved = count(roots, |row| status(row, CONVERGED))
                + count(extra, |row| row["eigenvalue_velocity_response"].is_string());
            return shape(roots.len() + extra.len(), resolved, 0, "point_measurement");
        }
    }
    if let (Some(norms), Some(vectors), true) = (
        data["residual_norms"].as_array(),
        data["eigenvectors"].as_array(),
        data.get("parity").is_some(),
    ) {
        let resolved = if vectors.len() == norms.len() {
            count(norms, |norm| norm.is_string())
        } else {
            0
        };
        return shape(norms.len(), resolved, 0, "point_measurement");
    }
    if data.get("rayleigh_quotient").is_some() {
        let ok = strings(
            data,
            &[
                "rayleigh_quotient",
                "relative_residual",
                "eigenvalue_defect",
            ],
        );
        return shape(1, usize::from(ok), 0, "point_measurement");
    }
    if data.get("unit_l2_center").is_some() {
        let ok = strings(
            data,
            &[
                "unit_l2_center",
                "coefficient_evenness_defect",
                "coefficient_norm",
            ],
        );
        return shape(1, usize::from(ok), 0, "point_measurement");
    }
    if let Some(values) = data["values"].as_object() {
        // Analyses that report scalar values instead of rows: each enclosed
        // value (with lower and upper bounds) is a row, else each value.
        let enclosed = values
            .keys()
            .filter(|key| {
                values.contains_key(&format!("{key}_lower"))
                    && values.contains_key(&format!("{key}_upper"))
            })
            .collect::<Vec<_>>();
        if !enclosed.is_empty() {
            let resolved = enclosed
                .iter()
                .filter(|key| {
                    [String::new(), "_lower".into(), "_upper".into()]
                        .iter()
                        .all(|suffix| values[&format!("{key}{suffix}")].is_string())
                })
                .count();
            return shape(enclosed.len(), resolved, 0, "point_measurement");
        }
        if !values.is_empty() {
            let resolved = values.values().filter(|value| value.is_string()).count();
            return shape(values.len(), resolved, 0, "point_measurement");
        }
    }
    if data.get("certifies_finite_ground_state_simple").is_some() {
        let certified = data["certifies_finite_ground_state_simple"].as_bool() == Some(true)
            && data["certifies_finite_matrix_positive_definite"].as_bool() == Some(true);
        return shape(1, usize::from(certified), 0, "certified_finite_enclosure");
    }
    None
}

#[derive(Clone)]
pub struct CapturedDiagnostic {
    pub value: serde_json::Value,
    /// Exact manifests supplied by the diagnostic's authenticated cache route.
    pub sources: Vec<ArtifactManifest>,
    /// When set, `value` is exactly the decoded payload of these artifacts
    /// (one artifact, or an ordered array when the flag is true). The receipt
    /// then records the references instead of a second copy of the data.
    pub value_reference: Option<(Vec<ArtifactManifest>, bool)>,
}
impl CapturedDiagnostic {
    /// A measurement whose value is exactly the decoded payload of the given
    /// artifacts: one artifact, or an ordered array of payloads.
    pub fn by_reference<T: Serialize>(
        value: &T,
        artifacts: Vec<ArtifactManifest>,
        array: bool,
        mut sources: Vec<ArtifactManifest>,
    ) -> Result<Self, CacheError> {
        if artifacts.is_empty() || (!array && artifacts.len() != 1) {
            return Err(invalid(
                "measurement reference requires its value artifacts",
            ));
        }
        for artifact in &artifacts {
            if !sources.contains(artifact) {
                sources.push(artifact.clone());
            }
        }
        let mut diagnostic = Self::new(value, sources)?;
        diagnostic.value_reference = Some((artifacts, array));
        Ok(diagnostic)
    }
    /// A retained qualified-absence record must not become a completed measurement.
    pub fn qualified(self) -> Result<Self, CaptureFailure> {
        if matches!(
            self.value["kind"].as_str(),
            Some(
                "ccm_compactness_analysis"
                    | "ccm_weighted_reference_projection"
                    | "ccm_signed_transform_analysis"
                    | "ccm_arithmetic_energy_analysis"
                    | "ccm_directional_response_analysis"
                    | "ccm_weighted_tail_analysis"
                    | "ccm_spectral_cluster_analysis"
                    | "ccm_resolution_budget_analysis"
                    | "ccm_energy_allowance_analysis"
                    | "ccm_complex_transform_analysis"
                    | "ccm_root_transport_analysis"
                    | "ccm_operator_cluster_analysis"
                    | "ccm_finite_section_transfer"
                    | "ccm_tail_operator_analysis"
                    | "ccm_observable_budget_analysis"
                    | "ccm_reference_projection_analysis"
                    | "ccm_capture_preflight"
                    | "ccm_consistency_analysis"
                    | "ccm_configuration_comparison"
                    | "ccm_band_reconstruction"
                    | "ccm_transform_enclosure"
            )
        ) && self.value["data"]["outcome"] == "missing_input"
        {
            return Err(CaptureFailure::Missing {
                reason: self.value["data"]["reason"]
                    .as_str()
                    .unwrap_or("required retained inputs unavailable")
                    .to_string(),
            });
        }
        Ok(self)
    }
    pub fn new<T: Serialize>(
        value: &T,
        sources: Vec<ArtifactManifest>,
    ) -> Result<Self, CacheError> {
        let value = xc_core::finite_json::to_value(value).map_err(|e| invalid(e.to_string()))?;
        xc_core::validate_secret_free(&value, "capture measurement")
            .map_err(|e| invalid(e.to_string()))?;
        Ok(Self {
            value,
            sources,
            value_reference: None,
        })
    }
    /// The measurement is the retained artifact itself; receipts reference it.
    pub fn from_cached<T: Serialize>(
        result: ArtifactExecutionCacheResult<T>,
    ) -> Result<Self, CacheError> {
        match result.produced_manifest.or(result.reused_manifest) {
            Some(manifest) => {
                Self::by_reference(&result.value, vec![manifest.clone()], false, vec![manifest])
            }
            None => Self::new(&result.value, Vec::new()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapturedMeasurement {
    /// Embedded value; null when the value is recorded by reference.
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub value: serde_json::Value,
    /// Artifacts whose decoded payload is the value (see `measurement_value`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value_reference: Option<MeasurementValueReference>,
    pub source_dependencies: Vec<DependencyRef>,
}

/// A measurement value stored once in the artifact fabric rather than copied
/// into the receipt. The digest binds the exact referenced value, and the
/// coverage summary is computed from it at capture time.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasurementValueReference {
    pub artifacts: Vec<DependencyRef>,
    /// True when the value is the ordered array of the artifacts' payloads.
    pub array: bool,
    pub value_digest: String,
    pub coverage: NumericalCoverage,
}

/// Recover a measurement value, reading referenced artifacts through
/// `resolver` and checking the bound digest.
pub fn measurement_value(
    measurement: &CapturedMeasurement,
    resolver: &CacheResolver,
    policy: &CachePolicy,
) -> Result<serde_json::Value, CacheError> {
    let Some(reference) = &measurement.value_reference else {
        return Ok(measurement.value.clone());
    };
    let mut values = Vec::with_capacity(reference.artifacts.len());
    for artifact in &reference.artifacts {
        let bytes = resolver.read_exact_payload(&artifact.key, &artifact.content_digest, policy)?;
        values.push(serde_json::from_slice::<serde_json::Value>(&bytes)?);
    }
    let value = if reference.array {
        serde_json::Value::Array(values)
    } else {
        values
            .pop()
            .ok_or_else(|| invalid("measurement reference has no artifact"))?
    };
    if xc_core::research_digest(&value)
        .map_err(|e| invalid(e.to_string()))?
        .0
        != reference.value_digest
    {
        return Err(invalid(
            "referenced measurement value does not match its digest",
        ));
    }
    Ok(value)
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureArtifact {
    pub schema_version: u32,
    pub semantics: String,
    pub resolved_plan: serde_json::Value,
    pub requested_diagnostics: Vec<String>,
    pub receipt: CaptureReceipt,
    pub measurements: BTreeMap<String, CapturedMeasurement>,
    pub source_dependencies: Vec<DependencyRef>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub numerical_coverage: BTreeMap<String, NumericalCoverage>,
}

fn canonical_dependencies(
    dependencies: impl IntoIterator<Item = DependencyRef>,
) -> Result<Vec<DependencyRef>, CacheError> {
    let mut ordered = BTreeMap::new();
    for d in dependencies {
        if d.key.kind.trim().is_empty()
            || d.key.logical_key.trim().is_empty()
            || !d.key.parameters_digest.validate()
            || !d.content_digest.validate()
        {
            return Err(invalid("invalid research dependency identity"));
        }
        let key = (
            d.key.kind.clone(),
            d.key.logical_key.clone(),
            d.key.parameters_digest.clone(),
            d.content_digest.clone(),
        );
        if ordered.insert(key, d.clone()).is_some_and(|old| old != d) {
            return Err(invalid("conflicting research dependency requirements"));
        }
    }
    Ok(ordered.into_values().collect())
}
/// The caller authenticates source manifests through its existing resolver.
/// This function validates their structure and retains their exact identities.
pub fn research_source_dependencies(
    manifests: &[ArtifactManifest],
) -> Result<Vec<DependencyRef>, CacheError> {
    let mut dependencies = Vec::new();
    for m in manifests {
        m.validate()?;
        if !m.quality.satisfies(CacheQuality::Validated) {
            return Err(invalid("research source quality is below validated"));
        }
        dependencies.push(DependencyRef {
            key: m.key.clone(),
            content_digest: m.content_digest.clone(),
            required_quality: CacheQuality::Validated,
        });
    }
    canonical_dependencies(dependencies)
}
/// Value-reference artifacts in their value order: an array value is the
/// ordered list of these payloads, so they are never canonically re-sorted.
fn ordered_value_references(
    manifests: &[ArtifactManifest],
) -> Result<Vec<DependencyRef>, CacheError> {
    let mut references = Vec::with_capacity(manifests.len());
    for m in manifests {
        let reference = research_source_dependencies(std::slice::from_ref(m))?
            .pop()
            .ok_or_else(|| invalid("measurement reference has no artifact"))?;
        if references.contains(&reference) {
            return Err(invalid("measurement reference repeats an artifact"));
        }
        references.push(reference);
    }
    Ok(references)
}
fn measurement_evidence(
    id: &str,
    measurement: &CapturedMeasurement,
) -> Result<EvidenceRef, CacheError> {
    Ok(EvidenceRef {
        kind: "captured_measurement".into(),
        identifier: id.into(),
        digest: Some(
            xc_core::research_digest(measurement)
                .map_err(|e| invalid(e.to_string()))?
                .0,
        ),
        description:
            "Embedded measurement and exact recorded source dependencies; no assurance upgrade"
                .into(),
    })
}
impl CaptureArtifact {
    /// Also works on historical receipts that do not embed a coverage summary.
    pub fn coverage(&self) -> BTreeMap<String, NumericalCoverage> {
        self.receipt.outcomes().iter().map(|(id, outcome)| {
            let summary = if let Some(m) = self.measurements.get(id) {
                m.value_reference.as_ref().map_or_else(
                    || NumericalCoverage::from_value(&m.value),
                    |reference| reference.coverage.clone(),
                )
            } else {
                let (status, reason) = match outcome {
                    DiagnosticOutcome::Missing { reason } => ("missing_input", reason.clone()),
                    DiagnosticOutcome::Blocked { reason } => ("blocked", reason.clone()),
                    DiagnosticOutcome::Failed { reason } => ("failed", reason.clone()),
                    _ => ("unassessed", "no retained measurement".into()),
                };
                NumericalCoverage { outcome: status.into(), reason: Some(reason), recovery: Some("recover this diagnostic from retained sources; inspect prerequisite or error".into()), ..Default::default() }
            };
            (id.clone(), summary)
        }).collect()
    }
    pub fn validate(&self) -> Result<(), CacheError> {
        if self.schema_version != 1 || self.semantics != CAPTURE_RECORD_SEMANTICS {
            return Err(invalid("unsupported capture record"));
        }
        let expected = CaptureReceipt::new(&self.resolved_plan, self.requested_diagnostics.clone())
            .map_err(|e| invalid(e.to_string()))?;
        self.receipt
            .validate_against(&expected)
            .map_err(|e| invalid(e.to_string()))?;
        let mut completed = BTreeSet::new();
        for (id, outcome) in self.receipt.outcomes() {
            match outcome {
                DiagnosticOutcome::Pending => {
                    return Err(invalid("capture record contains unaccounted work"))
                }
                DiagnosticOutcome::Completed { evidence } => {
                    let m = self
                        .measurements
                        .get(id)
                        .ok_or_else(|| invalid("completed capture measurement missing"))?;
                    if *evidence != vec![measurement_evidence(id, m)?] {
                        return Err(invalid(
                            "capture evidence digest or source binding mismatch",
                        ));
                    }
                    if canonical_dependencies(m.source_dependencies.clone())?
                        != m.source_dependencies
                    {
                        return Err(invalid("noncanonical measurement dependencies"));
                    }
                    if let Some(reference) = &m.value_reference {
                        if !m.value.is_null()
                            || reference.artifacts.is_empty()
                            || (!reference.array && reference.artifacts.len() != 1)
                            || reference
                                .artifacts
                                .iter()
                                .any(|artifact| !m.source_dependencies.contains(artifact))
                        {
                            return Err(invalid("invalid referenced measurement value"));
                        }
                    }
                    completed.insert(id);
                }
                _ => {}
            }
        }
        if completed != self.measurements.keys().collect() {
            return Err(invalid(
                "unrequested or unsuccessful measurement embedded in receipt",
            ));
        }
        let deps = canonical_dependencies(
            self.measurements
                .values()
                .flat_map(|m| m.source_dependencies.clone()),
        )?;
        if deps != self.source_dependencies {
            return Err(invalid("capture dependency closure mismatch"));
        }
        if !self.numerical_coverage.is_empty() && self.numerical_coverage != self.coverage() {
            return Err(invalid("capture numerical coverage mismatch"));
        }
        xc_core::validate_secret_free(self, "capture record").map_err(|e| invalid(e.to_string()))
    }
}

/// A replacement for one completed measurement, bound to its exact old evidence
/// digest. The caller authenticates and numerically repairs the source artifact.
/// This API rebuilds the receipt without turning failed acquisition into success.
pub struct CaptureMeasurementRepair {
    pub diagnostic: String,
    pub original_evidence_digest: String,
    pub replacement: CapturedMeasurement,
}

/// Create an additive capture attempt with corrected measurements and source
/// identities. The original plan, requested diagnostics and all non-completed
/// outcomes remain unchanged. The original receipt must be retained.
pub fn repair_capture_measurements(
    original: &CaptureArtifact,
    repairs: Vec<CaptureMeasurementRepair>,
) -> Result<CaptureArtifact, CacheError> {
    original.validate()?;
    let mut repaired = original.clone();
    let mut seen = BTreeSet::new();
    for repair in repairs {
        if !seen.insert(repair.diagnostic.clone()) {
            return Err(invalid("duplicate capture measurement repair"));
        }
        let old = original
            .measurements
            .get(&repair.diagnostic)
            .ok_or_else(|| invalid("repair requires a completed original measurement"))?;
        if measurement_evidence(&repair.diagnostic, old)?
            .digest
            .as_deref()
            != Some(repair.original_evidence_digest.as_str())
        {
            return Err(invalid("capture repair original evidence digest mismatch"));
        }
        if canonical_dependencies(repair.replacement.source_dependencies.clone())?
            != repair.replacement.source_dependencies
        {
            return Err(invalid("noncanonical capture repair dependencies"));
        }
        repaired
            .measurements
            .insert(repair.diagnostic, repair.replacement);
    }
    repaired.receipt = CaptureReceipt::new(
        &original.resolved_plan,
        original.requested_diagnostics.clone(),
    )
    .map_err(|e| invalid(e.to_string()))?;
    for (id, outcome) in original.receipt.outcomes() {
        let outcome = match outcome {
            DiagnosticOutcome::Completed { .. } => DiagnosticOutcome::Completed {
                evidence: vec![measurement_evidence(id, &repaired.measurements[id])?],
            },
            other => other.clone(),
        };
        repaired
            .receipt
            .record(id, outcome)
            .map_err(|e| invalid(e.to_string()))?;
    }
    repaired.source_dependencies = canonical_dependencies(
        repaired
            .measurements
            .values()
            .flat_map(|m| m.source_dependencies.clone()),
    )?;
    if !original.numerical_coverage.is_empty() {
        repaired.numerical_coverage = repaired.coverage();
    }
    repaired.validate()?;
    Ok(repaired)
}

/// Execute every requested diagnostic exactly once in deterministic order.
/// Returned failures are recorded and do not prevent independent diagnostics.
/// Invalid executor output becomes a failure with a secret-screened reason. Panics
/// and process termination are not converted into successful capture records.
pub fn collect_capture<P, F>(
    plan: &P,
    requested: Vec<String>,
    mut execute: F,
) -> Result<CaptureArtifact, CacheError>
where
    P: Serialize,
    F: FnMut(&str) -> Result<CapturedDiagnostic, CaptureFailure>,
{
    let resolved_plan = xc_core::finite_json::to_value(plan).map_err(|e| invalid(e.to_string()))?;
    xc_core::validate_secret_free(&resolved_plan, "capture plan")
        .map_err(|e| invalid(e.to_string()))?;
    let mut receipt = CaptureReceipt::new(&resolved_plan, requested.clone())
        .map_err(|e| invalid(e.to_string()))?;
    let mut measurements = BTreeMap::new();
    for id in &requested {
        let outcome = match execute(id).and_then(CapturedDiagnostic::qualified) {
            Ok(diagnostic) => {
                let measurement = (|| {
                    xc_core::validate_secret_free(&diagnostic.value, "capture measurement")
                        .map_err(|e| invalid(e.to_string()))?;
                    let source_dependencies = research_source_dependencies(&diagnostic.sources)?;
                    let m = match &diagnostic.value_reference {
                        None => CapturedMeasurement {
                            value: diagnostic.value,
                            value_reference: None,
                            source_dependencies,
                        },
                        Some((artifacts, array)) => CapturedMeasurement {
                            value_reference: Some(MeasurementValueReference {
                                artifacts: ordered_value_references(artifacts)?,
                                array: *array,
                                value_digest: xc_core::research_digest(&diagnostic.value)
                                    .map_err(|e| invalid(e.to_string()))?
                                    .0,
                                coverage: NumericalCoverage::from_value(&diagnostic.value),
                            }),
                            value: serde_json::Value::Null,
                            source_dependencies,
                        },
                    };
                    let evidence = vec![measurement_evidence(id, &m)?];
                    Ok::<_, CacheError>((m, evidence))
                })();
                match measurement {
                    Ok((m, evidence)) => {
                        measurements.insert(id.clone(), m);
                        DiagnosticOutcome::Completed { evidence }
                    }
                    Err(error) => DiagnosticOutcome::Failed {
                        reason: safe_failure_reason(error),
                    },
                }
            }
            Err(failure) => failure.outcome(),
        };
        if receipt.record(id, outcome).is_err() {
            receipt
                .record(
                    id,
                    DiagnosticOutcome::Failed {
                        reason: "diagnostic returned an invalid outcome; details omitted".into(),
                    },
                )
                .map_err(|e| invalid(e.to_string()))?;
        }
    }
    let source_dependencies = canonical_dependencies(
        measurements
            .values()
            .flat_map(|m| m.source_dependencies.clone()),
    )?;
    let mut record = CaptureArtifact {
        schema_version: 1,
        semantics: CAPTURE_RECORD_SEMANTICS.into(),
        resolved_plan,
        requested_diagnostics: requested,
        receipt,
        measurements,
        source_dependencies,
        numerical_coverage: BTreeMap::new(),
    };
    record.numerical_coverage = record.coverage();
    record.validate()?;
    Ok(record)
}

/// Persist a fully identified research record. Numerical child APIs perform
/// their own reuse; this operation preserves each distinct receipt/evaluation
/// rather than allowing an incomplete earlier attempt to suppress new work.
/// Warm reads are validated against the supplied record and exact dependencies.
pub fn persist_research_artifact<T, V>(
    kind: &str,
    record: &T,
    dependencies: &[DependencyRef],
    cache: &ArtifactCacheContext<'_>,
    validate: V,
) -> Result<ArtifactExecutionCacheResult<T>, CacheError>
where
    T: Serialize + DeserializeOwned + Clone,
    V: Fn(&T) -> Result<(), CacheError>,
{
    if !matches!(kind, CAPTURE_RECEIPT_KIND | HYPOTHESIS_EVALUATION_KIND) {
        return Err(invalid("unsupported managed research record kind"));
    }
    if cache.write_visibility == CacheVisibility::Public {
        return Err(invalid("full research records are private-only"));
    }
    if cache.requested_assurance != xc_core::AssuranceLevel::Computed {
        return Err(invalid(
            "research records do not upgrade numerical assurance",
        ));
    }
    validate(record)?;
    xc_core::validate_secret_free(record, "managed research record")
        .map_err(|e| invalid(e.to_string()))?;
    let dependencies = canonical_dependencies(dependencies.to_vec())?;
    let record_digest = xc_core::research_digest(record).map_err(|e| invalid(e.to_string()))?;
    let mut tags = BTreeMap::from([("assurance".into(), "computed_not_certified".into())]);
    let logical = if kind == CAPTURE_RECEIPT_KIND {
        let capture: CaptureArtifact = serde_json::from_value(
            serde_json::to_value(record).map_err(|e| invalid(e.to_string()))?,
        )
        .map_err(|e| invalid(e.to_string()))?;
        capture.validate()?;
        let plan = capture.receipt.plan_digest();
        tags.insert("research_plan_digest".into(), plan.0.clone());
        format!(
            "{}{}/{}",
            capture_receipt_plan_prefix(plan)?,
            "attempt",
            record_digest.0
        )
    } else {
        format!("research/{kind}/{}", record_digest.0)
    };
    let semantic = SemanticKeyEnvelope {
        schema_version: 1,
        artifact_kind: kind.into(),
        mathematical_semantics_version: "managed-research-record-v1".into(),
        resolved_mathematical_parameters: serde_json::json!({"record_digest": record_digest, "source_dependencies": dependencies}),
        normalization: None,
        target: Some("finite_research_evidence".into()),
        subspace: None,
        source_data_identities: BTreeMap::new(),
        algorithm_semantics: Some("recorded_outcomes_no_assurance_upgrade_v1".into()),
    };
    let request = ArtifactExecutionCacheRequest {
        operation: "research.record.persist",
        semantic_key: &semantic,
        logical_key: &logical,
        resolver: cache.resolver,
        reference_resolver: cache.reference_resolver,
        acceptance: cache.acceptance,
        ordered_overlays: cache.ordered_overlays.clone(),
        mode: cache.mode,
        write_on_miss: cache.write_on_miss,
        write_visibility: cache.write_visibility,
        produced_quality: CacheQuality::Validated,
        producer_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION"))?,
        minimum_reader_version: ToolkitVersion::parse(crate::CLEAN_SLATE)?,
        maximum_reader_version: None,
        tags,
        provenance_digest: None,
        production_sink: cache.production_sink,
    };
    let result = resolve_or_compute_json_artifact_with_dependencies(
        &request,
        || Ok((record.clone(), dependencies.clone())),
        |cached: &T| {
            validate(cached)?;
            if xc_core::research_digest(cached).map_err(|e| invalid(e.to_string()))?
                != record_digest
            {
                return Err(invalid("research record identity mismatch"));
            }
            Ok(())
        },
    )?;
    if let Some(manifest) = result
        .produced_manifest
        .as_ref()
        .or(result.reused_manifest.as_ref())
    {
        validate_managed_dependencies(manifest, &semantic, &dependencies, cache)?;
    }
    Ok(result)
}

fn validate_managed_dependencies(
    manifest: &ArtifactManifest,
    semantic: &SemanticKeyEnvelope,
    dependencies: &[DependencyRef],
    cache: &ArtifactCacheContext<'_>,
) -> Result<(), CacheError> {
    let Some(encoded) = manifest.tags.get(REMOTE_CANONICAL_MANIFEST_TAG) else {
        return if manifest.dependencies == dependencies {
            Ok(())
        } else {
            Err(invalid("managed research dependency closure mismatch"))
        };
    };
    // Shard adapters deliberately have no key-based dependency list. Validate
    // their authenticated canonical closure instead, including exact source
    // content and quality. An empty adapter list is not proof of an empty graph.
    if !manifest.dependencies.is_empty() && manifest.dependencies != dependencies {
        return Err(invalid("managed research dependency closure mismatch"));
    }
    let canonical: CanonicalArtifactManifest = serde_json::from_str(encoded)?;
    let provenance = manifest
        .provenance_digest
        .as_ref()
        .ok_or_else(|| invalid("managed research canonical provenance missing"))?;
    validate_retained_canonical_binding(
        &canonical,
        semantic,
        "ccm-evidence",
        &manifest.content_digest,
        manifest.size_bytes,
        Some(provenance),
    )?;
    let identity = |dependency: &DependencyRef| {
        (
            dependency.key.kind.clone(),
            dependency.key.parameters_digest.clone(),
            dependency.content_digest.clone(),
        )
    };
    // Logical aliases are local lookup names. Canonical publication deduplicates
    // them when they name the same mathematical artifact and content.
    let expected: BTreeSet<_> = dependencies.iter().map(&identity).collect();
    if canonical.canonical_payload.dependencies.len() != expected.len() {
        return Err(invalid(
            "managed research canonical dependency count mismatch",
        ));
    }
    let mut seen = BTreeSet::new();
    for declared in &canonical.canonical_payload.dependencies {
        let resolver = cache
            .resolver
            .ok_or_else(|| invalid("managed research canonical closure lacks resolver"))?;
        let policy = cache
            .acceptance
            .ok_or_else(|| invalid("managed research canonical closure lacks policy"))?;
        let (_, source) = resolver
            .resolve_dependency_identity_manifest(declared, policy)?
            .ok_or_else(|| {
                invalid(format!(
                    "managed research canonical dependency unavailable: {}/{}",
                    declared.artifact_family, declared.semantic_digest
                ))
            })?;
        let actual = (
            source.key.kind.clone(),
            source.key.parameters_digest.clone(),
            source.content_digest.clone(),
        );
        if !expected.contains(&actual) || !seen.insert(actual.clone()) {
            return Err(invalid(
                "managed research canonical source identity mismatch",
            ));
        }
        for dependency in dependencies.iter().filter(|d| identity(d) == actual) {
            if !source.quality.satisfies(dependency.required_quality) {
                return Err(invalid(
                    "managed research canonical source quality mismatch",
                ));
            }
        }
    }
    Ok(())
}

/// Prefix for grouping distinct attempts of one resolved plan. This matches
/// manifest logical keys and the `research_plan_digest` manifest tag; terminal
/// failures remain separate attempts and are never overwritten by later work.
pub fn capture_receipt_plan_prefix(plan: &xc_core::ConfigDigest) -> Result<String, CacheError> {
    if !ContentDigest(plan.0.clone()).validate() {
        return Err(invalid("invalid capture plan digest"));
    }
    Ok(format!("research/{CAPTURE_RECEIPT_KIND}/{}/", plan.0))
}

pub fn persist_capture(
    record: &CaptureArtifact,
    cache: &ArtifactCacheContext<'_>,
) -> Result<ArtifactExecutionCacheResult<CaptureArtifact>, CacheError> {
    persist_research_artifact(
        CAPTURE_RECEIPT_KIND,
        record,
        &record.source_dependencies,
        cache,
        CaptureArtifact::validate,
    )
}

pub fn capture_and_persist<P, F>(
    plan: &P,
    requested: Vec<String>,
    execute: F,
    cache: &ArtifactCacheContext<'_>,
) -> Result<ArtifactExecutionCacheResult<CaptureArtifact>, CacheError>
where
    P: Serialize,
    F: FnMut(&str) -> Result<CapturedDiagnostic, CaptureFailure>,
{
    if cache.write_visibility == CacheVisibility::Public {
        return Err(invalid("full research records are private-only"));
    }
    let record = collect_capture(plan, requested, execute)?;
    persist_capture(&record, cache)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn capture_accepts_validated_and_stronger_sources_without_assurance_upgrade() {
        let bytes = b"synthetic source";
        let digest = ContentDigest::sha256(bytes);
        let mut source = ArtifactManifest {
            schema_version: 1,
            key: ArtifactKey::new("synthetic", "quality-regression", b"quality").unwrap(),
            content_digest: digest.clone(),
            size_bytes: bytes.len() as u64,
            objects: vec![CacheObjectRef {
                content_digest: digest,
                size_bytes: bytes.len() as u64,
            }],
            created_unix_seconds: 1,
            producer_toolkit_version: ToolkitVersion::parse("0.18.0").unwrap(),
            minimum_reader_version: ToolkitVersion::parse("0.16.0").unwrap(),
            maximum_reader_version: None,
            quality: CacheQuality::Validated,
            visibility: CacheVisibility::Local,
            immutable: true,
            dependencies: vec![],
            tags: BTreeMap::new(),
            provenance_digest: None,
        };
        for quality in [
            CacheQuality::Validated,
            CacheQuality::CrossChecked,
            CacheQuality::Certified,
            CacheQuality::Published,
            CacheQuality::Staged,
            CacheQuality::Quarantined,
            CacheQuality::Deprecated,
        ] {
            source.quality = quality;
            let record = collect_capture(&json!({}), vec!["source".into(), "next".into()], |id| {
                CapturedDiagnostic::new(
                    &json!(42),
                    if id == "source" {
                        vec![source.clone()]
                    } else {
                        vec![]
                    },
                )
                .map_err(CaptureFailure::failed)
            })
            .unwrap();
            record.validate().unwrap();
            assert!(record.measurements.contains_key("next"));
            if quality.satisfies(CacheQuality::Validated) {
                assert!(record.receipt.is_complete(), "{quality:?}");
                assert_eq!(
                    record.source_dependencies[0].required_quality,
                    CacheQuality::Validated
                );
                assert_eq!(
                    record.source_dependencies[0].content_digest,
                    source.content_digest
                );
            } else {
                let DiagnosticOutcome::Failed { reason } = &record.receipt.outcomes()["source"]
                else {
                    panic!("inadmissible source accepted")
                };
                assert!(reason.contains("quality is below validated"));
            }
        }
    }

    #[test]
    fn capture_errors_preserve_safe_details_and_screen_secrets() {
        let reason = "prefix working precision must be at least source precision";
        assert_eq!(
            CaptureFailure::failed(reason),
            CaptureFailure::Failed {
                reason: reason.into()
            }
        );
        // SECRET_AUDIT_PATTERN: intentionally invalid credential-bearing error fixture.
        let secret = format!("{}{}", "ghp_", "a".repeat(36));
        let encoded = serde_json::to_string(&CaptureFailure::failed(&secret)).unwrap();
        assert!(!encoded.contains(&secret));
        assert!(encoded.contains("details omitted"));
    }

    fn capture() -> CaptureArtifact {
        collect_capture(
            &json!({"dimension":32}),
            vec!["a".into(), "b".into(), "c".into(), "d".into()],
            |id| match id {
                "a" => CapturedDiagnostic::new(&json!({"numerically_resolved":false}), vec![])
                    .map_err(|_| unreachable!()),
                "b" => Err(CaptureFailure::Missing {
                    reason: "source absent".into(),
                }),
                "c" => Err(CaptureFailure::Failed {
                    reason: "solver returned an error".into(),
                }),
                _ => Err(CaptureFailure::Blocked {
                    reason: "dimension budget".into(),
                }),
            },
        )
        .unwrap()
    }

    #[test]
    fn retained_capture_repair_preserves_failures_and_rebinds_evidence() {
        let original = capture();
        let saved = original.clone();
        let old = original.measurements["a"].clone();
        let replacement = CapturedMeasurement {
            value: json!({"corrected": 17}),
            value_reference: None,
            source_dependencies: vec![],
        };
        let repair = || CaptureMeasurementRepair {
            diagnostic: "a".into(),
            original_evidence_digest: xc_core::research_digest(&old).unwrap().0,
            replacement: replacement.clone(),
        };
        let corrected = repair_capture_measurements(&original, vec![repair()]).unwrap();
        assert_eq!(original, saved);
        assert_eq!(corrected.resolved_plan, original.resolved_plan);
        assert_eq!(
            corrected.requested_diagnostics,
            original.requested_diagnostics
        );
        assert_eq!(
            corrected.receipt.plan_digest(),
            original.receipt.plan_digest()
        );
        assert_eq!(corrected.measurements["a"], replacement);
        assert_ne!(
            corrected.receipt.outcomes()["a"],
            original.receipt.outcomes()["a"]
        );
        for id in ["b", "c", "d"] {
            assert_eq!(
                corrected.receipt.outcomes()[id],
                original.receipt.outcomes()[id]
            );
        }
        assert!(!corrected.receipt.is_complete());
        assert!(repair_capture_measurements(&original, vec![repair(), repair()]).is_err());
        let mut invalid = repair();
        invalid.original_evidence_digest = "0".repeat(64);
        assert!(repair_capture_measurements(&original, vec![invalid]).is_err());
        let mut invalid = repair();
        invalid.diagnostic = "b".into();
        assert!(repair_capture_measurements(&original, vec![invalid]).is_err());
        let mut tampered = corrected;
        tampered.measurements.get_mut("a").unwrap().value = json!(99);
        assert!(tampered.validate().is_err());
    }
    #[test]
    fn outcomes_preserve_partial_acquisition_without_promoting_numerical_success() {
        let c = capture();
        c.validate().unwrap();
        assert_eq!(c.receipt.outcomes().len(), 4);
        assert!(!c.receipt.is_complete());
        assert_eq!(c.measurements["a"].value["numerically_resolved"], false);
        assert!(matches!(
            c.receipt.outcomes()["d"],
            DiagnosticOutcome::Blocked { .. }
        ));
        assert!(
            collect_capture(&json!({}), vec!["x".into(), "x".into()], |_| panic!(
                "duplicate plan must not execute"
            ))
            .is_err()
        );
    }
    #[test]
    fn receipt_rejects_tampered_bytes_dropped_outcomes_and_invented_work() {
        let original = capture();
        let mut c = original.clone();
        c.measurements.get_mut("a").unwrap().value = json!({"numerically_resolved":true});
        assert!(c.validate().is_err());
        let mut c = original.clone();
        c.measurements
            .insert("b".into(), c.measurements["a"].clone());
        assert!(c.validate().is_err());
        let mut encoded = serde_json::to_value(&original).unwrap();
        encoded["receipt"]["outcomes"]
            .as_object_mut()
            .unwrap()
            .remove("b");
        assert!(serde_json::from_value::<CaptureArtifact>(encoded)
            .unwrap()
            .validate()
            .is_err());
        let mut encoded = serde_json::to_value(&original).unwrap();
        encoded["receipt"]["outcomes"]["b"] = json!({"status":"pending"});
        assert!(serde_json::from_value::<CaptureArtifact>(encoded)
            .unwrap()
            .validate()
            .is_err());
    }
    #[test]
    fn invalid_executor_outcomes_are_redacted_without_dropping_later_work() {
        let c = collect_capture(&json!({}), vec!["bad".into(), "next".into()], |id| {
            if id == "bad" {
                Err(CaptureFailure::Failed { reason: "".into() })
            } else {
                CapturedDiagnostic::new(&json!(42), vec![]).map_err(|_| unreachable!())
            }
        })
        .unwrap();
        assert!(matches!(
            c.receipt.outcomes()["bad"],
            DiagnosticOutcome::Failed { .. }
        ));
        assert!(c.measurements.contains_key("next"));
    }
    #[test]
    fn both_research_kinds_are_private_in_every_routing_policy() {
        for kind in [CAPTURE_RECEIPT_KIND, HYPOTHESIS_EVALUATION_KIND] {
            assert_eq!(family_for_artifact_kind(kind), Some("ccm-evidence"));
            assert!(artifact_kind_is_private_only(kind));
            assert!(!artifact_kind_admitted_to_destination(
                kind,
                PublicationDestination::Public
            ));
            assert!(artifact_kind_admitted_to_destination(
                kind,
                PublicationDestination::Private
            ));
            let policy = artifact_compatibility_policy("ccm-evidence", kind).unwrap();
            assert_eq!(
                policy.minimum_reader_version,
                ToolkitVersion::parse(crate::CLEAN_SLATE).unwrap()
            );
        }
    }
    #[test]
    fn referenced_measurements_are_not_copied_and_reconstruct_exactly() {
        let root = crate::test_support::temporary_root("referenced-measurements");
        let store = FilesystemCacheStore::new("record", root.path(), true, CacheVisibility::Local);
        let put = |name: &str, value: &serde_json::Value| {
            let draft = ArtifactDraft {
                schema_version: 1,
                key: ArtifactKey::new("synthetic", name, name.as_bytes()).unwrap(),
                producer_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION")).unwrap(),
                minimum_reader_version: ToolkitVersion::parse("0.16.0").unwrap(),
                maximum_reader_version: None,
                quality: CacheQuality::Validated,
                visibility: CacheVisibility::Local,
                immutable: true,
                dependencies: Vec::new(),
                tags: BTreeMap::new(),
                provenance_digest: None,
            };
            store
                .put(&draft, &serde_json::to_vec(value).unwrap())
                .unwrap()
        };
        let single = json!({"rows":[{"outcome":"point_measurement"},{"outcome":"unresolved"}]});
        let first = json!({"value": 1});
        let second = json!({"value": [2, 3]});
        let single_manifest = put("single", &single);
        let first_manifest = put("first", &first);
        let second_manifest = put("second", &second);
        let record = collect_capture(
            &json!({}),
            vec!["array".into(), "reversed".into(), "single".into()],
            |id| {
                if id == "reversed" {
                    // Value order differs from canonical dependency order.
                    CapturedDiagnostic::by_reference(
                        &json!([second.clone(), first.clone()]),
                        vec![second_manifest.clone(), first_manifest.clone()],
                        true,
                        vec![],
                    )
                } else if id == "single" {
                    CapturedDiagnostic::by_reference(
                        &single,
                        vec![single_manifest.clone()],
                        false,
                        vec![],
                    )
                } else {
                    CapturedDiagnostic::by_reference(
                        &json!([first.clone(), second.clone()]),
                        vec![first_manifest.clone(), second_manifest.clone()],
                        true,
                        vec![],
                    )
                }
                .map_err(CaptureFailure::failed)
            },
        )
        .unwrap();
        record.validate().unwrap();
        assert!(record.receipt.is_complete());
        assert!(record.measurements.values().all(|m| m.value.is_null()));
        assert_eq!(record.numerical_coverage["single"].resolved_rows, 1);
        assert_eq!(record.numerical_coverage["single"].unresolved_rows, 1);
        let resolver = CacheResolver::new(vec![CacheLayer {
            precedence: 0,
            store: Box::new(store),
        }]);
        let policy = CachePolicy {
            current_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION")).unwrap(),
            minimum_quality: CacheQuality::Validated,
            accepted_schema_versions: vec![1],
            allow_deprecated: false,
            allow_quarantined: false,
            allowed_visibilities: vec![CacheVisibility::Local],
        };
        assert_eq!(
            measurement_value(&record.measurements["single"], &resolver, &policy).unwrap(),
            single
        );
        assert_eq!(
            measurement_value(&record.measurements["array"], &resolver, &policy).unwrap(),
            json!([first, second])
        );
        assert_eq!(
            measurement_value(&record.measurements["reversed"], &resolver, &policy).unwrap(),
            json!([second, first])
        );
        let mut tampered = record.measurements["single"].clone();
        tampered.value_reference.as_mut().unwrap().value_digest = "0".repeat(64);
        assert!(measurement_value(&tampered, &resolver, &policy).is_err());
        let mut embedded = record.clone();
        embedded.measurements.get_mut("single").unwrap().value = json!(1);
        assert!(embedded.validate().is_err());
        let _ = std::fs::remove_dir_all(root);
    }
    #[test]
    fn managed_receipts_reuse_exact_attempts_and_preserve_changed_outcomes() {
        let scratch = crate::test_support::TestDir::new("research-record");
        let root = scratch.join("root");
        let resolver = CacheResolver::new(vec![CacheLayer {
            precedence: 0,
            store: Box::new(FilesystemCacheStore::new(
                "record",
                root.clone(),
                true,
                CacheVisibility::Local,
            )),
        }]);
        let policy = CachePolicy {
            current_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION")).unwrap(),
            minimum_quality: CacheQuality::Validated,
            accepted_schema_versions: vec![1],
            allow_deprecated: false,
            allow_quarantined: false,
            allowed_visibilities: vec![CacheVisibility::Local],
        };
        let context = |mode: ArtifactExecutionCacheMode| ArtifactCacheContext {
            resolver: Some(&resolver),
            reference_resolver: None,
            acceptance: Some(&policy),
            ordered_overlays: vec!["record".into()],
            mode,
            write_on_miss: !mode.requires_reuse(),
            write_visibility: CacheVisibility::Local,
            requested_assurance: xc_core::AssuranceLevel::Computed,
            certification_failure_policy: CertificationFailurePolicy::RetainComputedFailRun,
            production_sink: None,
        };
        let c = capture();
        assert!(persist_capture(&c, &context(ArtifactExecutionCacheMode::RequireReuse)).is_err());
        let cold = persist_capture(&c, &context(ArtifactExecutionCacheMode::PreferReuse)).unwrap();
        let warm = persist_capture(&c, &context(ArtifactExecutionCacheMode::RequireReuse)).unwrap();
        assert_eq!(cold.value, warm.value);
        assert_eq!(cold.produced_manifest, warm.reused_manifest);
        let manifest = cold.produced_manifest.as_ref().unwrap();
        assert!(manifest
            .key
            .logical_key
            .starts_with(&capture_receipt_plan_prefix(c.receipt.plan_digest()).unwrap()));
        assert_eq!(
            manifest.tags["research_plan_digest"],
            c.receipt.plan_digest().0
        );
        let changed = collect_capture(
            &json!({"dimension":32}),
            c.requested_diagnostics.clone(),
            |_| {
                CapturedDiagnostic::new(&json!("now available"), vec![]).map_err(|_| unreachable!())
            },
        )
        .unwrap();
        assert!(changed.receipt.is_complete());
        assert!(
            persist_capture(&changed, &context(ArtifactExecutionCacheMode::RequireReuse)).is_err()
        );
        let newer =
            persist_capture(&changed, &context(ArtifactExecutionCacheMode::PreferReuse)).unwrap();
        assert_ne!(
            cold.produced_manifest.unwrap().key,
            newer.produced_manifest.unwrap().key
        );
        let mut public = context(ArtifactExecutionCacheMode::PreferReuse);
        public.write_visibility = CacheVisibility::Public;
        assert!(capture_and_persist(
            &json!({}),
            vec!["a".into()],
            |_| panic!("public request must fail before work"),
            &public
        )
        .is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
}

#[cfg(test)]
mod coverage_tests {
    use super::*;
    #[test]
    fn acquired_rows_are_not_automatically_resolved() {
        let c = NumericalCoverage::from_value(
            &serde_json::json!({"request":{"expected_rows":4},"data":{"outcome":"point_measurement","rows":[{"outcome":"point_measurement"},{"outcome":"unresolved_denominator"},{"outcome":"conditional_budget_met"}]}}),
        );
        assert_eq!(c.retained_rows, 3);
        assert_eq!(c.expected_rows, Some(4));
        assert_eq!(c.resolved_rows, 1);
        assert_eq!(c.qualified_rows, 1);
        assert_eq!(c.unresolved_rows, 1);
        assert_eq!(c.outcome, "partial_unresolved");
        assert!(c.recovery.is_some());
        let legacy = NumericalCoverage::from_value(&serde_json::json!({"value":"3"}));
        assert_eq!(legacy.outcome, "unassessed");
        assert_eq!(legacy.expected_rows, None);
    }

    #[test]
    fn capture_payload_shapes_have_numerical_coverage() {
        use serde_json::json;
        let cov = |v: serde_json::Value| {
            let c = NumericalCoverage::from_value(&v);
            (
                c.outcome,
                c.retained_rows,
                c.resolved_rows,
                c.qualified_rows,
                c.unresolved_rows,
            )
        };
        let o = |s: &str| s.to_owned();
        assert_eq!(
            cov(
                json!({"evenness_deviation":"1e-9","natural_eigenvalue":"3","forced_eigenvalue":"3"})
            ),
            (o("point_measurement"), 1, 1, 0, 0)
        );
        let spectrum = json!({"parity":"even","eigenvectors":[[],[]],"residual_norms":["1e-90","1e-80"],"eigenvalues":["1","2"]});
        let gap = json!({"gap_log":"5","even_simple":true,"lambda_even":"1","lambda_odd":"2"});
        assert_eq!(
            cov(json!([gap, spectrum])),
            (o("point_measurement"), 3, 3, 0, 0)
        );
        let checkpoint =
            |status: &str| json!({"dimension":121,"squared_overlap":"1","status":status});
        assert_eq!(
            cov(checkpoint("export_checks_passed")),
            (o("point_measurement"), 1, 1, 0, 0)
        );
        assert_eq!(
            cov(checkpoint(
                "innovation_export_passed_eigenpair_not_supplied"
            ))
            .3,
            1
        );
        assert_eq!(
            cov(checkpoint("eigenpair_mismatch")),
            (o("partial_unresolved"), 1, 0, 0, 1)
        );
        let ladder =
            json!({"ladder":{"stopped":null},"checkpoints":[checkpoint("export_checks_passed")]});
        assert_eq!(cov(ladder), (o("point_measurement"), 2, 2, 0, 0));
        let stopped = json!({"ladder":{"stopped":"precision"},"checkpoints":[checkpoint("export_checks_passed")]});
        assert_eq!(cov(stopped).4, 1);
        assert_eq!(
            cov(json!({"checks_passed":false,"computed_eigenvalues":[]})).4,
            1
        );
        let band = json!({"data":{"points":[{"source_status":"converged"}],"missing_count":1}});
        assert_eq!(cov(band), (o("partial_unresolved"), 2, 1, 0, 1));
        let conditioning =
            json!([{"root_count":2,"outcomes":[{"status":"converged"},{"status":"stagnated"}]}]);
        assert_eq!(cov(conditioning).4, 1);
        let response = json!([{"roots":[{"status":"converged"}],"events":[{"eigenvalue_velocity_response":"2"}]}]);
        assert_eq!(cov(response), (o("point_measurement"), 2, 2, 0, 0));
        let flow = json!([{"roots":[{"status":"converged"}],"channels":[{"eigenvalue_velocity_response":"2"},{}]}]);
        assert_eq!(cov(flow).4, 1);
        assert_eq!(cov(json!({"data":{"rayleigh_quotient":"3","relative_residual":"1e-90","eigenvalue_defect":"0"}})).0, "point_measurement");
        assert_eq!(cov(json!({"unit_l2_center":"1","coefficient_evenness_defect":"0","coefficient_norm":"1"})).0, "point_measurement");
        let certificate = |simple: bool| json!({"certifies_finite_ground_state_simple":simple,"certifies_finite_matrix_positive_definite":true});
        assert_eq!(
            cov(certificate(true)),
            (o("certified_finite_enclosure"), 1, 1, 0, 0)
        );
        assert_eq!(cov(certificate(false)).4, 1);
        assert_eq!(
            cov(json!({"levels":[{},{}],"outcome":"computed"})),
            (o("point_measurement"), 2, 2, 0, 0)
        );
        assert_eq!(
            cov(json!({"levels":[{}],"outcome":"computed_with_limitation"})),
            (o("qualified"), 1, 0, 1, 0)
        );
        let assembly = json!({"full":{"spectral_upper":"1e-981"},"even_sector":{"spectral_upper":"1e-981"},
            "odd_sector":{},"exact_form_bounds":[{"exact_form_lower":"1","exact_form_upper":"2"}]});
        assert_eq!(cov(assembly), (o("partial_unresolved"), 4, 3, 0, 1));
        let overlap = json!({"outcome":"point_measurement","signed_unit_overlap":"0.9","difference_norm_squared":"0.1","rhs":[],"coefficients":[]});
        assert_eq!(cov(overlap), (o("point_measurement"), 1, 1, 0, 0));
        let unsolved = json!({"outcome":"rank_or_precision_unresolved","signed_unit_overlap":"0.9","difference_norm_squared":"0.1","rhs":["1","2"],"coefficients":null});
        assert_eq!(cov(unsolved), (o("partial_unresolved"), 3, 1, 0, 2));
        let enclosed = json!({"outcome":"point_measurement","rows":[],"values":{"w":"1","w_lower":"0.9","w_upper":"1.1","bits":"64"}});
        assert_eq!(cov(enclosed), (o("point_measurement"), 1, 1, 0, 0));
        let empty =
            NumericalCoverage::from_value(&json!({"outcome":"point_measurement","rows":[]}));
        assert_eq!(
            (empty.outcome.as_str(), empty.retained_rows),
            ("unassessed", 0)
        );
        let rows = json!({"data":{"rows":[{"outcome":"point_measurement"}]}});
        assert_eq!(cov(rows), (o("point_measurement"), 1, 1, 0, 0));
    }

    #[test]
    fn distance_payloads_have_numerical_coverage() {
        use serde_json::json;
        let profile = NumericalCoverage::from_value(
            &json!({"u_values":["1","2"],"f_values":["1","0.5"],"sample_count":2}),
        );
        assert_eq!(
            (
                profile.outcome.as_str(),
                profile.retained_rows,
                profile.resolved_rows
            ),
            ("point_measurement", 2, 2)
        );
        let distance = NumericalCoverage::from_value(&json!({"measurements":[
            {"distance_to_target":"0.1","eigenfunction_norm":"1"},{"distance_to_target":"0.2","eigenfunction_norm":"1"}]}));
        assert_eq!(
            (distance.outcome.as_str(), distance.resolved_rows),
            ("point_measurement", 2)
        );
        let residual = NumericalCoverage::from_value(
            &json!({"measurements":[{"absolute_residual_mass":"3"}]}),
        );
        assert_eq!(residual.outcome, "point_measurement");
        let deviation = NumericalCoverage::from_value(&json!({"projections":[
            {"amplitude":"1","relative_residual":"0.1"},{"amplitude":"2","relative_residual":"0.2"}]}));
        assert_eq!(
            (deviation.outcome.as_str(), deviation.resolved_rows),
            ("point_measurement", 2)
        );
        let evidence = |row_met: bool, ladder: bool| {
            json!({
            "reported_resolution_tolerance_met": true, "refinement_ladder_tolerance_met": ladder,
            "retained_evidence": [{"refinements":[{"tolerance_met":true},{"tolerance_met":row_met}]}]})
        };
        let met = NumericalCoverage::from_value(&evidence(true, true));
        assert_eq!(
            (met.outcome.as_str(), met.retained_rows, met.resolved_rows),
            ("point_measurement", 4, 4)
        );
        let row_failed = NumericalCoverage::from_value(&evidence(false, true));
        assert_eq!(
            (row_failed.outcome.as_str(), row_failed.unresolved_rows),
            ("partial_unresolved", 1)
        );
        let ladder_failed = NumericalCoverage::from_value(&evidence(true, false));
        assert_eq!(
            (
                ladder_failed.outcome.as_str(),
                ladder_failed.unresolved_rows
            ),
            ("partial_unresolved", 1)
        );
        // Other shapes are unchanged.
        let other = NumericalCoverage::from_value(&json!({"measurements":[{"value":"1"}]}));
        assert_eq!(other.outcome, "unassessed");
    }
    #[test]
    fn summaries_are_validated_and_historical_receipts_remain_readable() {
        let r=collect_capture(&serde_json::json!({"policy":"test"}),vec!["a".into()],|_|CapturedDiagnostic::new(&serde_json::json!({"data":{"outcome":"partial_unresolved","reason":"no slope"}}),vec![]).map_err(CaptureFailure::failed)).unwrap();
        assert_eq!(r.numerical_coverage["a"].outcome, "partial_unresolved");
        let mut corrupted = r.clone();
        corrupted.numerical_coverage.get_mut("a").unwrap().outcome = "point_measurement".into();
        assert!(corrupted.validate().is_err());
        let mut historical = serde_json::to_value(r).unwrap();
        historical
            .as_object_mut()
            .unwrap()
            .remove("numerical_coverage");
        serde_json::from_value::<CaptureArtifact>(historical)
            .unwrap()
            .validate()
            .unwrap();
    }
}

#[cfg(test)]
mod enclosure_coverage_audit {
    #[test]
    fn enclosure_rows_remain_distinct_from_point_samples() {
        let c = super::NumericalCoverage::from_value(
            &serde_json::json!({"data":{"outcome":"partial_unresolved","rows":[{"outcome":"certified_finite_enclosure"},{"outcome":"point_measurement"},{"outcome":"missing_input"}]}}),
        );
        assert_eq!(c.resolved_rows, 2);
        assert_eq!(c.unresolved_rows, 1);
        assert_eq!(c.row_outcomes["certified_finite_enclosure"], 1);
        assert_eq!(c.row_outcomes["point_measurement"], 1);
    }
}
