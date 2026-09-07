// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Interpretation, frozen predictions and resolution contracts for retained evidence.
//! Definitions and model coefficients are supplied by callers, never built into
//! the numerical library. A frozen digest proves identity, not prior blindness.
use crate::config_resolution::{canonical_json, sha256_hex};
use crate::{ArtifactRef, ConfigDigest, ConfigError, DecimalLiteral, EvidenceRef};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const HYPOTHESIS_SCORING_SEMANTICS: &str = "frozen-finite-domain-absolute-envelope-v1";

fn required(value: &str, field: &str) -> Result<(), ConfigError> {
    if value.trim().is_empty() {
        return Err(ConfigError::new(format!("{field} must be nonempty")));
    }
    Ok(())
}
fn digest(value: &ConfigDigest, field: &str) -> Result<(), ConfigError> {
    if !value.is_sha256() {
        return Err(ConfigError::new(format!(
            "{field} must be lowercase SHA-256"
        )));
    }
    Ok(())
}
/// Canonical, secret-free JSON identity shared by frozen plans and managed records.
pub fn research_digest<T: Serialize>(value: &T) -> Result<ConfigDigest, ConfigError> {
    crate::validate_secret_free(value, "research specification")?;
    let value = serde_json::to_value(value).map_err(|e| ConfigError::new(e.to_string()))?;
    let text = canonical_json(&value).map_err(|e| ConfigError::new(e.to_string()))?;
    Ok(ConfigDigest(sha256_hex(text.as_bytes())))
}
fn nonnegative(value: &DecimalLiteral) -> Result<(), ConfigError> {
    if value.cmp_numeric(&DecimalLiteral::new("0")?)?.is_lt() {
        return Err(ConfigError::new("resolution/remainder must be nonnegative"));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservableTarget {
    Finite,
    StabilizedPlateau,
    SampledMinimum,
    ExtrapolatedLimit,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogBase {
    Natural,
    Decimal,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum ObservableTransform {
    Identity,
    PositiveLog { base: LogBase, negative: bool },
    SignedLogMagnitude { base: LogBase, negative: bool },
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RootErrorConvention {
    Absolute,
    Relative,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RootObservable {
    /// One-based reference ordinal; acquisition method lives in the design.
    pub index: usize,
    pub reference_identity: ConfigDigest,
    pub error: RootErrorConvention,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProlateObservable {
    pub index: usize,
    pub indexing_convention: String,
    pub deficiency_convention: String,
    pub asymptotic_approximation: bool,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservableContract {
    pub definition_id: String,
    pub definition_version: u32,
    pub target: ObservableTarget,
    pub transform: ObservableTransform,
    pub normalization: String,
    pub metric: String,
    pub target_definition: Option<ConfigDigest>,
    pub derivative_coordinate: Option<String>,
    pub root: Option<RootObservable>,
    pub prolate: Option<ProlateObservable>,
}
impl ObservableContract {
    pub fn validate(&self) -> Result<(), ConfigError> {
        required(&self.definition_id, "observable definition")?;
        required(&self.normalization, "observable normalization")?;
        required(&self.metric, "observable metric")?;
        if self.definition_version == 0 {
            return Err(ConfigError::new("observable version must be positive"));
        }
        if let Some(d) = &self.target_definition {
            digest(d, "target definition")?;
        }
        if let Some(c) = &self.derivative_coordinate {
            required(c, "derivative coordinate")?;
        }
        if let Some(r) = &self.root {
            if r.index == 0 {
                return Err(ConfigError::new("root index must be one-based"));
            }
            digest(&r.reference_identity, "root reference")?;
        }
        if let Some(p) = &self.prolate {
            required(&p.indexing_convention, "prolate indexing convention")?;
            required(&p.deficiency_convention, "prolate deficiency convention")?;
        }
        Ok(())
    }
    /// Algorithms may differ in a planned cross-check. Mathematical definitions
    /// may not: conversions require a new explicit derived observation.
    pub fn check_compatible(&self, other: &Self) -> Result<(), ConfigError> {
        self.validate()?;
        other.validate()?;
        macro_rules! same { ($($f:ident),+) => { $(if self.$f != other.$f {
            return Err(ConfigError::new(concat!("incompatible observable: ", stringify!($f))));
        })+ }; }
        same!(
            definition_id,
            definition_version,
            target,
            transform,
            normalization,
            metric,
            target_definition,
            derivative_coordinate,
            root,
            prolate
        );
        Ok(())
    }
}

/// An exact planned design, including algorithmic differences. A cohort is an
/// explicit list of these designs, not a grouping by ambiguous column names.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationDesign {
    /// Names include the convention, e.g. "c=lambda^2" or "u=ln(c)".
    pub coordinates: BTreeMap<String, DecimalLiteral>,
    pub n_modes: Option<usize>,
    pub dimension: usize,
    pub parity: String,
    pub basis: String,
    pub method: String,
    pub quadrature_identity: String,
    pub source_precision_bits: u32,
    pub exact_input_identity: ConfigDigest,
}
impl ObservationDesign {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.dimension == 0 || self.source_precision_bits < 2 || self.coordinates.is_empty() {
            return Err(ConfigError::new(
                "observation requires dimension, source precision and exact coordinates",
            ));
        }
        for (k, v) in &self.coordinates {
            required(k, "coordinate")?;
            v.validate()?;
        }
        for (k, v) in [
            ("parity", &self.parity),
            ("basis", &self.basis),
            ("method", &self.method),
            ("quadrature identity", &self.quadrature_identity),
        ] {
            required(v, k)?;
        }
        digest(&self.exact_input_identity, "exact input identity")
    }
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationMetadata {
    pub case_id: String,
    /// Independent design unit, normally an entire cutoff family.
    pub family_id: String,
    pub observable: ObservableContract,
    pub design: ObservationDesign,
    pub payload_sha256: ConfigDigest,
    pub sources: Vec<ArtifactRef>,
}
impl ObservationMetadata {
    pub fn validate(&self) -> Result<(), ConfigError> {
        required(&self.case_id, "case ID")?;
        required(&self.family_id, "family ID")?;
        self.observable.validate()?;
        self.design.validate()?;
        digest(&self.payload_sha256, "observation payload")?;
        if self.sources.is_empty() {
            return Err(ConfigError::new("observation requires source artifacts"));
        }
        for s in &self.sources {
            required(&s.kind, "source kind")?;
            required(&s.logical_key, "source key")?;
            digest(&ConfigDigest(s.semantic_digest.clone()), "source semantics")?;
            digest(&ConfigDigest(s.payload_digest.clone()), "source payload")?;
        }
        crate::validate_secret_free(self, "observation metadata")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResolutionAxis {
    SourceConstruction,
    WorkingArithmetic,
    Solve,
    FiniteN,
    MeasurementQuadrature,
    Reference,
    ExportRoundtrip,
}
pub const RESOLUTION_AXES: [ResolutionAxis; 7] = [
    ResolutionAxis::SourceConstruction,
    ResolutionAxis::WorkingArithmetic,
    ResolutionAxis::Solve,
    ResolutionAxis::FiniteN,
    ResolutionAxis::MeasurementQuadrature,
    ResolutionAxis::Reference,
    ResolutionAxis::ExportRoundtrip,
];
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResolutionClass {
    RigorousBound,
    ConditionalBound,
    EmpiricalDiscrepancy,
    Estimate,
    Unknown,
    NotApplicable,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResolutionComponent {
    pub axis: ResolutionAxis,
    pub classification: ResolutionClass,
    /// Absolute uncertainty in the observable's declared units, not input units.
    pub absolute: Option<DecimalLiteral>,
    pub explanation: String,
    pub evidence: Vec<EvidenceRef>,
    /// Shared parent/group identifiers; the scorer never assumes independence.
    pub dependency_groups: BTreeSet<String>,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservableResolution {
    pub components: Vec<ResolutionComponent>,
    pub source_precision_bits: u32,
    pub analysis_precision_bits: u32,
    pub export_significant_digits: usize,
}
impl ObservableResolution {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.source_precision_bits < 2
            || self.analysis_precision_bits < 2
            || self.export_significant_digits == 0
        {
            return Err(ConfigError::new(
                "source, analysis and export precision must be explicit",
            ));
        }
        let mut axes = BTreeSet::new();
        for c in &self.components {
            if !axes.insert(c.axis) {
                return Err(ConfigError::new("duplicate resolution axis"));
            }
            required(&c.explanation, "resolution explanation")?;
            let absent = matches!(
                c.classification,
                ResolutionClass::Unknown | ResolutionClass::NotApplicable
            );
            if absent != c.absolute.is_none() {
                return Err(ConfigError::new(
                    "unknown/not-applicable has no numeric bound; other components require one",
                ));
            }
            if let Some(x) = &c.absolute {
                nonnegative(x)?;
            }
            if !absent && c.evidence.is_empty() {
                return Err(ConfigError::new(
                    "numerical resolution needs evidence provenance",
                ));
            }
            if matches!(
                c.classification,
                ResolutionClass::RigorousBound | ResolutionClass::ConditionalBound
            ) && c.evidence.iter().all(|e| {
                e.digest
                    .as_ref()
                    .is_none_or(|d| !ConfigDigest(d.clone()).is_sha256())
            }) {
                return Err(ConfigError::new(
                    "declared bounds need digest-bound evidence",
                ));
            }
        }
        if axes.len() != RESOLUTION_AXES.len() {
            return Err(ConfigError::new(
                "all seven resolution axes must be explicit, including unknown/not-applicable",
            ));
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NonzeroSign {
    Positive,
    Negative,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "state")]
pub enum ObservedScalar {
    SignedLogMagnitude {
        sign: NonzeroSign,
        log_magnitude: DecimalLiteral,
    },
    Finite {
        value: DecimalLiteral,
    },
    ExactZero,
    RoundedZero,
    UnresolvedSign {
        approximation: DecimalLiteral,
    },
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationPayload {
    pub schema_version: u32,
    /// Repeated deliberately to bind metadata selection to the decoded bytes.
    pub observable: ObservableContract,
    pub design: ObservationDesign,
    pub value: ObservedScalar,
    pub resolution: ObservableResolution,
}
impl ObservationPayload {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.schema_version != 1 {
            return Err(ConfigError::new("unsupported observation schema"));
        }
        self.observable.validate()?;
        self.design.validate()?;
        self.resolution.validate()?;
        if self.design.source_precision_bits != self.resolution.source_precision_bits {
            return Err(ConfigError::new(
                "source precision disagrees with the resolution record",
            ));
        }
        let signed_log = matches!(
            self.observable.transform,
            ObservableTransform::SignedLogMagnitude { .. }
        );
        match &self.value {
            ObservedScalar::ExactZero if signed_log => {
                return Err(ConfigError::new(
                    "an exact original zero has no finite signed logarithm",
                ))
            }
            ObservedScalar::SignedLogMagnitude { log_magnitude, .. } => {
                if !signed_log {
                    return Err(ConfigError::new(
                        "signed-log value requires a signed-log observable",
                    ));
                }
                log_magnitude.validate()?;
            }
            ObservedScalar::Finite { value } => {
                if signed_log {
                    return Err(ConfigError::new(
                        "signed-log observable requires an explicit original sign",
                    ));
                }
                value.validate()?;
            }
            ObservedScalar::UnresolvedSign { approximation } => approximation.validate()?,
            _ => {}
        }
        crate::validate_secret_free(self, "observation payload")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetPartition {
    Calibration,
    Development,
    Replication,
    ProtectedValidation,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "origin")]
pub enum ParameterOrigin {
    TheoryFixed { justification: String },
    External { evidence: String },
    Fitted { calibration_digest: ConfigDigest },
    PostHoc { selection_record: String },
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HypothesisParameter {
    pub value: DecimalLiteral,
    pub origin: ParameterOrigin,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlannedObservation {
    pub prediction_sign: Option<NonzeroSign>,
    pub family_id: String,
    pub design: ObservationDesign,
    pub partition: DatasetPartition,
    /// Frozen prediction in the observable's own units. No runtime expression eval.
    pub prediction: DecimalLiteral,
    /// The stated finite-domain absolute tolerance/remainder, including model
    /// evaluation error where necessary. Not inferred from an asymptotic symbol.
    pub absolute_tolerance: DecimalLiteral,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HypothesisSpec {
    pub schema_version: u32,
    pub scoring_semantics: String,
    pub model_id: String,
    pub model_version: String,
    pub evaluator_digest: ConfigDigest,
    pub observable: ObservableContract,
    pub parameters: BTreeMap<String, HypothesisParameter>,
    pub regime: String,
    pub selection_policy: String,
    pub exposure_record: String,
    /// Includes prior failures/selection when known. Saving a hash is not a
    /// prospective timestamp and no prospective flag is synthesized here.
    pub prior_specification: Option<ConfigDigest>,
    pub exclusions: BTreeMap<String, String>,
    pub cases: BTreeMap<String, PlannedObservation>,
}
impl HypothesisSpec {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.schema_version != 1
            || self.scoring_semantics != HYPOTHESIS_SCORING_SEMANTICS
            || self.cases.is_empty()
        {
            return Err(ConfigError::new(
                "hypothesis requires schema 1 and planned cases",
            ));
        }
        for (k, v) in [
            ("model", &self.model_id),
            ("model version", &self.model_version),
            ("regime", &self.regime),
            ("selection policy", &self.selection_policy),
            ("exposure record", &self.exposure_record),
        ] {
            required(v, k)?;
        }
        self.observable.validate()?;
        digest(&self.evaluator_digest, "evaluator")?;
        if let Some(d) = &self.prior_specification {
            digest(d, "prior specification")?;
        }
        for (name, param) in &self.parameters {
            required(name, "parameter name")?;
            param.value.validate()?;
            match &param.origin {
                ParameterOrigin::TheoryFixed { justification } => {
                    required(justification, "theory justification")?
                }
                ParameterOrigin::External { evidence } => {
                    required(evidence, "external parameter evidence")?
                }
                ParameterOrigin::Fitted { calibration_digest } => {
                    digest(calibration_digest, "calibration")?
                }
                ParameterOrigin::PostHoc { selection_record } => {
                    required(selection_record, "post-hoc selection")?
                }
            }
        }
        let mut families = BTreeMap::new();
        let mut designs = BTreeMap::new();
        for (id, c) in &self.cases {
            required(id, "case ID")?;
            required(&c.family_id, "family ID")?;
            c.design.validate()?;
            let design_digest = research_digest(&c.design)?.0;
            if designs
                .insert(design_digest, (c.partition, c.family_id.clone()))
                .is_some_and(|(p, family)| p != c.partition || family != c.family_id)
            {
                return Err(ConfigError::new(
                    "the same exact design cannot be split across partitions or independent families",
                ));
            }
            if c.prediction_sign.is_some()
                != matches!(
                    self.observable.transform,
                    ObservableTransform::SignedLogMagnitude { .. }
                )
            {
                return Err(ConfigError::new(
                    "prediction sign must be explicit exactly for signed-log observables",
                ));
            }
            c.prediction.validate()?;
            nonnegative(&c.absolute_tolerance)?;
            if self.exclusions.contains_key(id) {
                return Err(ConfigError::new("planned case is also excluded"));
            }
            if families
                .insert(&c.family_id, c.partition)
                .is_some_and(|p| p != c.partition)
            {
                return Err(ConfigError::new(
                    "a design family cannot straddle fitting and validation partitions",
                ));
            }
        }
        for (id, reason) in &self.exclusions {
            required(id, "exclusion ID")?;
            required(reason, "exclusion reason")?;
        }
        crate::validate_secret_free(self, "hypothesis")
    }
    pub fn freeze(self) -> Result<FrozenHypothesis, ConfigError> {
        self.validate()?;
        let digest = research_digest(&self)?;
        Ok(FrozenHypothesis {
            specification: self,
            digest,
        })
    }
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenHypothesis {
    specification: HypothesisSpec,
    digest: ConfigDigest,
}
impl FrozenHypothesis {
    pub fn specification(&self) -> &HypothesisSpec {
        &self.specification
    }
    pub fn digest(&self) -> &ConfigDigest {
        &self.digest
    }
    pub fn validate(&self) -> Result<(), ConfigError> {
        self.specification.validate()?;
        if research_digest(&self.specification)? != self.digest {
            return Err(ConfigError::new("frozen hypothesis digest mismatch"));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HypothesisVerdict {
    PassOnTestedDomain,
    FailOnTestedDomain,
    UnresolvedAtCurrentResolution,
    IneligibleData,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "status")]
pub enum DiagnosticOutcome {
    Pending,
    Completed { evidence: Vec<EvidenceRef> },
    Missing { reason: String },
    Blocked { reason: String },
    Failed { reason: String },
}
/// A receipt accounts for requested work; it cannot start an acquisition or
/// promote numerical assurance. Requested IDs cannot be silently dropped.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureReceipt {
    plan_digest: ConfigDigest,
    outcomes: BTreeMap<String, DiagnosticOutcome>,
}
impl CaptureReceipt {
    pub fn new<T: Serialize>(
        plan: &T,
        requested: impl IntoIterator<Item = String>,
    ) -> Result<Self, ConfigError> {
        let mut outcomes = BTreeMap::new();
        for id in requested {
            required(&id, "diagnostic ID")?;
            if outcomes.insert(id, DiagnosticOutcome::Pending).is_some() {
                return Err(ConfigError::new("duplicate requested diagnostic"));
            }
        }
        Ok(Self {
            plan_digest: research_digest(plan)?,
            outcomes,
        })
    }
    pub fn plan_digest(&self) -> &ConfigDigest {
        &self.plan_digest
    }
    pub fn outcomes(&self) -> &BTreeMap<String, DiagnosticOutcome> {
        &self.outcomes
    }
    pub fn record(&mut self, id: &str, outcome: DiagnosticOutcome) -> Result<(), ConfigError> {
        match &outcome {
            DiagnosticOutcome::Completed { evidence } if evidence.is_empty() => {
                return Err(ConfigError::new("completed capture requires evidence"))
            }
            DiagnosticOutcome::Missing { reason }
            | DiagnosticOutcome::Blocked { reason }
            | DiagnosticOutcome::Failed { reason } => {
                required(reason, "diagnostic outcome reason")?
            }
            _ => {}
        }
        crate::validate_secret_free(&outcome, "capture outcome")?;
        let entry = self
            .outcomes
            .get_mut(id)
            .ok_or_else(|| ConfigError::new("diagnostic was not requested by this receipt"))?;
        if !matches!(entry, DiagnosticOutcome::Pending) {
            return Err(ConfigError::new(
                "capture outcome already recorded; create a new receipt for another attempt",
            ));
        }
        *entry = outcome;
        Ok(())
    }
    /// Compare with a freshly resolved expected receipt on import: serde alone
    /// cannot establish that a producer retained every requested diagnostic.
    pub fn validate_against(&self, expected: &Self) -> Result<(), ConfigError> {
        if self.plan_digest != expected.plan_digest
            || self.outcomes.keys().ne(expected.outcomes.keys())
        {
            return Err(ConfigError::new(
                "capture receipt does not account for the expected plan",
            ));
        }
        let mut replay = expected.clone();
        for (id, outcome) in &self.outcomes {
            replay.record(id, outcome.clone())?;
        }
        Ok(())
    }
    pub fn is_complete(&self) -> bool {
        self.outcomes
            .values()
            .all(|s| matches!(s, DiagnosticOutcome::Completed { .. }))
    }
}
