// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! HP scoring of frozen finite-domain predictions against retained observations.
//! Payload selection precedes loading. No fitting, acquisition, publication or
//! source assurance upgrade occurs. Interval arithmetic controls analysis
//! rounding; declared empirical uncertainties remain empirical.
use crate::mpfr_interval::MpfrInterval;
use crate::prefix::lossless_decimal;
use anyhow::{bail, Result};
use rug::{float::Round, Float};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use xc_cache::ContentDigest;
use xc_core::*;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecimalEnclosure {
    pub lower: String,
    pub upper: String,
}
impl DecimalEnclosure {
    fn from_interval(value: &MpfrInterval) -> Self {
        // lossless_decimal round-trips in nearest rounding. Widen endpoints
        // once before export so exact decimal consumers retain the enclosure.
        let mut lower = value.lower().clone();
        lower.next_down();
        let mut upper = value.upper().clone();
        upper.next_up();
        Self {
            lower: lossless_decimal(&lower),
            upper: lossless_decimal(&upper),
        }
    }
}
fn precision(p: u32) -> Result<()> {
    if !(64..=1_000_000).contains(&p) {
        bail!("analysis precision must be 64..=1000000 bits");
    }
    Ok(())
}
/// Enclose the exact supplied decimal without a binary64 intermediate. This
/// preserves the supplied digits, not information absent from a rounded source.
pub fn enclose_decimal(value: &DecimalLiteral, p: u32) -> Result<MpfrInterval> {
    precision(p)?;
    value.validate()?;
    let lo = Float::with_val_round(p, Float::parse(value.as_str())?, Round::Down).0;
    let hi = Float::with_val_round(p, Float::parse(value.as_str())?, Round::Up).0;
    if !lo.is_finite()
        || !hi.is_finite()
        || (lo.is_zero() && hi.is_zero() && !value.cmp_numeric(&DecimalLiteral::new("0")?)?.is_eq())
    {
        bail!("decimal exceeds the analysis backend exponent range");
    }
    Ok(MpfrInterval::new(lo, hi)?)
}

/// Explicit log-unit conversion, including a depth sign change. The input is
/// already a logarithmic observable, not its underlying positive value.
pub fn convert_log_units(
    value: &DecimalLiteral,
    from: LogBase,
    from_negative: bool,
    to: LogBase,
    to_negative: bool,
    p: u32,
) -> Result<DecimalEnclosure> {
    let mut x = enclose_decimal(value, p)?;
    if from != to {
        let ln10 = MpfrInterval::from_i64(10, p).ln()?;
        x = if from == LogBase::Decimal {
            x.mul(&ln10)
        } else {
            x.div(&ln10)?
        };
    }
    if from_negative != to_negative {
        x = x.neg();
    }
    Ok(DecimalEnclosure::from_interval(&x))
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResolutionSummary {
    pub known_radius_upper: String,
    pub unknown_axes: Vec<ResolutionAxis>,
    pub classifications: Vec<ResolutionClass>,
    pub dependency_groups: BTreeSet<String>,
    /// Arithmetic sum in observable units; never root-sum-square.
    pub aggregation: String,
}
fn resolution_range(
    value: &ObservedScalar,
    resolution: &ObservableResolution,
    p: u32,
) -> Result<(Option<MpfrInterval>, ResolutionSummary)> {
    resolution.validate()?;
    let mut radius = MpfrInterval::from_i64(0, p);
    let mut summary = ResolutionSummary { known_radius_upper: String::new(), unknown_axes: vec![],
        classifications: vec![], dependency_groups: BTreeSet::new(),
        aggregation: "L1 sum of declared absolute components; empirical terms remain empirical; unknown is not zero".into() };
    for c in &resolution.components {
        summary.classifications.push(c.classification);
        summary
            .dependency_groups
            .extend(c.dependency_groups.iter().cloned());
        if c.classification == ResolutionClass::Unknown {
            summary.unknown_axes.push(c.axis);
        }
        if let Some(x) = &c.absolute {
            radius = radius.add(&enclose_decimal(x, p)?);
        }
    }
    summary.known_radius_upper = DecimalEnclosure::from_interval(&radius).upper;
    let center = match value {
        ObservedScalar::Finite { value } => Some(enclose_decimal(value, p)?),
        ObservedScalar::SignedLogMagnitude { log_magnitude, .. } => {
            Some(enclose_decimal(log_magnitude, p)?)
        }
        ObservedScalar::ExactZero => Some(MpfrInterval::from_i64(0, p)),
        ObservedScalar::RoundedZero | ObservedScalar::UnresolvedSign { .. } => None,
    };
    let interval = if summary.unknown_axes.is_empty() {
        match center {
            Some(x) => {
                let padding = MpfrInterval::new(-radius.upper().clone(), radius.upper().clone())?;
                let result = x.add(&padding);
                Some(MpfrInterval::new(
                    result.lower().clone(),
                    result.upper().clone(),
                )?)
            }
            None => None,
        }
    } else {
        None
    };
    Ok((interval, summary))
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaseScore {
    pub case_id: String,
    pub family_id: String,
    pub verdict: HypothesisVerdict,
    pub reason: String,
    pub source_payload: Option<ConfigDigest>,
    pub observed: Option<ObservedScalar>,
    pub resolution: Option<ObservableResolution>,
    pub resolution_summary: Option<ResolutionSummary>,
    pub signed_residual: Option<DecimalEnclosure>,
    pub prediction: DecimalLiteral,
    pub absolute_tolerance: DecimalLiteral,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HypothesisScore {
    pub semantics: String,
    pub specification_digest: ConfigDigest,
    pub partition: DatasetPartition,
    pub analysis_precision_bits: u32,
    pub verdict: HypothesisVerdict,
    /// Every required case in the requested partition produced an eligible
    /// payload. Complete acquisition can still be scientifically unresolved.
    pub complete: bool,
    pub required_observations: usize,
    pub loaded_observations: usize,
    pub design_families: usize,
    pub declared_parameters: usize,
    pub cases: Vec<CaseScore>,
    pub exclusions: BTreeMap<String, String>,
    pub interpretation: String,
}

/// The only call to `load` is after frozen case/partition, definition, design,
/// duplicate and metadata checks. The loader should read existing bytes only.
/// The callback API allows cache, local-file and in-memory adapters without a
/// second resolver. Protected validation requires explicit access on this call.
pub fn score_frozen_hypothesis<F>(
    frozen: &FrozenHypothesis,
    metadata: &[ObservationMetadata],
    partition: DatasetPartition,
    allow_protected_validation: bool,
    p: u32,
    mut load: F,
) -> Result<ResearchResult<HypothesisScore>>
where
    F: FnMut(&ObservationMetadata) -> Result<Vec<u8>>,
{
    frozen.validate()?;
    precision(p)?;
    if partition == DatasetPartition::ProtectedValidation && !allow_protected_validation {
        bail!("protected validation access was not authorized for this scoring call");
    }
    let spec = frozen.specification();
    let planned = spec
        .cases
        .iter()
        .filter(|(_, c)| c.partition == partition)
        .collect::<Vec<_>>();
    if planned.is_empty() {
        bail!("the requested partition has no frozen required cases");
    }
    let mut indexed: BTreeMap<&str, Vec<&ObservationMetadata>> = BTreeMap::new();
    let mut exclusions = spec.exclusions.clone();
    for m in metadata {
        match spec.cases.get(&m.case_id) {
            Some(c) if c.partition == partition => indexed.entry(&m.case_id).or_default().push(m),
            Some(_) => {
                exclusions.insert(
                    m.case_id.clone(),
                    "outside the requested partition; payload not read".into(),
                );
            }
            None => {
                exclusions
                    .entry(m.case_id.clone())
                    .or_insert_with(|| "not in frozen cohort; payload not read".into());
            }
        }
    }
    let protected_hashes = metadata
        .iter()
        .filter(|m| {
            spec.cases
                .get(&m.case_id)
                .is_some_and(|c| c.partition == DatasetPartition::ProtectedValidation)
        })
        .map(|m| m.payload_sha256.0.as_str())
        .collect::<BTreeSet<_>>();
    let mut payload_counts: BTreeMap<&str, usize> = BTreeMap::new();
    for group in indexed.values() {
        for m in group {
            *payload_counts.entry(&m.payload_sha256.0).or_default() += 1;
        }
    }
    let mut scores = vec![];
    let mut artifacts = vec![];
    let mut loaded = 0;
    for (id, c) in &planned {
        let mut score = CaseScore {
            case_id: (*id).clone(),
            family_id: c.family_id.clone(),
            verdict: HypothesisVerdict::UnresolvedAtCurrentResolution,
            reason: "required observation missing".into(),
            source_payload: None,
            observed: None,
            resolution: None,
            resolution_summary: None,
            signed_residual: None,
            prediction: c.prediction.clone(),
            absolute_tolerance: c.absolute_tolerance.clone(),
        };
        let Some(group) = indexed.get(id.as_str()) else {
            scores.push(score);
            continue;
        };
        if group.len() != 1 || payload_counts[group[0].payload_sha256.0.as_str()] != 1 {
            score.verdict = HypothesisVerdict::IneligibleData;
            score.reason = "duplicate case or payload; select one acquisition explicitly, not independent rows".into();
            scores.push(score);
            continue;
        }
        let m = group[0];
        let eligible = (|| -> Result<()> {
            m.validate()?;
            if partition != DatasetPartition::ProtectedValidation
                && protected_hashes.contains(m.payload_sha256.0.as_str())
            {
                bail!("selected payload aliases a protected outcome; payload not read");
            }
            spec.observable.check_compatible(&m.observable)?;
            if m.family_id != c.family_id || m.design != c.design {
                bail!("metadata differs from frozen design/family");
            }
            if m.sources
                .iter()
                .any(|s| s.completion != CompletionStatus::Successful)
            {
                bail!("source computation did not complete successfully");
            }
            Ok(())
        })();
        if let Err(e) = eligible {
            score.verdict = HypothesisVerdict::IneligibleData;
            score.reason = e.to_string();
            scores.push(score);
            continue;
        }
        let bytes = match load(m) {
            Ok(bytes) => bytes,
            Err(_) => {
                score.reason = "required retained payload unavailable (loader failed)".into();
                scores.push(score);
                continue;
            }
        };
        let decoded = (|| -> Result<ObservationPayload> {
            if ContentDigest::sha256(&bytes).0 != m.payload_sha256.0 {
                bail!("payload digest mismatch");
            }
            let payload: ObservationPayload = serde_json::from_slice(&bytes)
                .map_err(|_| anyhow::anyhow!("invalid observation payload encoding"))?;
            payload.validate()?;
            m.observable.check_compatible(&payload.observable)?;
            if payload.design != m.design {
                bail!("decoded payload differs from selected design");
            }
            Ok(payload)
        })();
        let payload = match decoded {
            Ok(x) => x,
            Err(e) => {
                score.verdict = HypothesisVerdict::IneligibleData;
                score.reason = e.to_string();
                scores.push(score);
                continue;
            }
        };
        loaded += 1;
        artifacts.extend(m.sources.iter().cloned());
        score.source_payload = Some(m.payload_sha256.clone());
        score.observed = Some(payload.value.clone());
        score.resolution = Some(payload.resolution.clone());
        let scored = (|| -> Result<()> {
            let (range, summary) = resolution_range(&payload.value, &payload.resolution, p)?;
            score.resolution_summary = Some(summary);
            let Some(range) = range else {
                score.reason =
                    "unknown resolution component or unresolved zero/sign; no resolved verdict"
                        .into();
                return Ok(());
            };
            if let ObservedScalar::SignedLogMagnitude { sign, .. } = &payload.value {
                if Some(*sign) != c.prediction_sign {
                    score.verdict = HypothesisVerdict::FailOnTestedDomain;
                    score.reason =
                        "resolved original sign disagrees with frozen signed-log prediction".into();
                    return Ok(());
                }
            }
            let residual = range.sub(&enclose_decimal(&c.prediction, p)?);
            let tolerance = enclose_decimal(&c.absolute_tolerance, p)?;
            // Use inward limits for pass and outward limits for fail; a
            // rounding-straddled boundary stays unresolved.
            if residual.lower() >= &(-tolerance.lower().clone())
                && residual.upper() <= tolerance.lower()
            {
                score.verdict = HypothesisVerdict::PassOnTestedDomain;
                score.reason = "entire declared observation range satisfies the frozen finite-domain tolerance".into();
            } else if residual.lower() > tolerance.upper()
                || residual.upper() < &(-tolerance.upper().clone())
            {
                score.verdict = HypothesisVerdict::FailOnTestedDomain;
                score.reason =
                    "declared observation range is disjoint from the frozen prediction tolerance"
                        .into();
            } else {
                score.reason =
                    "observation range overlaps the acceptance boundary; resolution cannot decide"
                        .into();
            }
            score.signed_residual = Some(DecimalEnclosure::from_interval(&residual));
            Ok(())
        })();
        if scored.is_err() {
            score.verdict = HypothesisVerdict::UnresolvedAtCurrentResolution;
            score.reason = "analysis arithmetic cannot resolve these supplied scales".into();
        }
        scores.push(score);
    }
    let complete = loaded == planned.len();
    let verdict = if scores
        .iter()
        .any(|s| s.verdict == HypothesisVerdict::FailOnTestedDomain)
    {
        HypothesisVerdict::FailOnTestedDomain
    } else if scores
        .iter()
        .any(|s| s.verdict == HypothesisVerdict::IneligibleData)
    {
        HypothesisVerdict::IneligibleData
    } else if complete
        && scores
            .iter()
            .all(|s| s.verdict == HypothesisVerdict::PassOnTestedDomain)
    {
        HypothesisVerdict::PassOnTestedDomain
    } else {
        HypothesisVerdict::UnresolvedAtCurrentResolution
    };
    let report = HypothesisScore {
        semantics: HYPOTHESIS_SCORING_SEMANTICS.into(),
        specification_digest: frozen.digest().clone(), partition, analysis_precision_bits: p,
        verdict, complete, required_observations: planned.len(), loaded_observations: loaded,
        design_families: planned.iter().map(|(_, c)| &c.family_id).collect::<BTreeSet<_>>().len(),
        declared_parameters: spec.parameters.len(), cases: scores, exclusions,
        interpretation: "Conditional on supplied definitions, source provenance and declared resolution. No statistical independence, prospective blindness, asymptotic theorem or source assurance upgrade is inferred.".into(),
    };
    let mut result = ResearchResult::computed(
        report,
        SolverProvenance::current_package("mpfr-directed-research-scoring"),
    );
    result.artifacts = artifacts;
    result.evidence.push(EvidenceRef {
        kind: "frozen_hypothesis".into(),
        identifier: spec.model_id.clone(),
        digest: Some(frozen.digest().0.clone()),
        description: "immutable specification and prediction table".into(),
    });
    result.validate_for_persistence()?;
    Ok(result)
}

/// Cancellation-aware (observed - leading) / correction scale. The supplied
/// leading term and scale are exact input decimals; their own model uncertainty
/// must already be included in the declared observable budget. No fit implied.
pub fn scaled_correction(
    payload: &ObservationPayload,
    leading: &DecimalLiteral,
    scale: &DecimalLiteral,
    p: u32,
) -> Result<Option<DecimalEnclosure>> {
    payload.validate()?;
    precision(p)?;
    if matches!(
        payload.observable.transform,
        ObservableTransform::SignedLogMagnitude { .. }
    ) {
        bail!("convert signed-log observations explicitly before linear correction analysis");
    }
    let scale = enclose_decimal(scale, p)?;
    if scale.contains_zero() {
        bail!("correction scale must be nonzero");
    }
    let (range, _) = resolution_range(&payload.value, &payload.resolution, p)?;
    range
        .map(|r| {
            Ok(DecimalEnclosure::from_interval(
                &r.sub(&enclose_decimal(leading, p)?).div(&scale)?,
            ))
        })
        .transpose()
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StabilizationStep {
    pub from_n_modes: usize,
    pub to_n_modes: usize,
    pub signed_change: Option<DecimalEnclosure>,
    pub ratio_denominator: Option<DecimalEnclosure>,
    pub signed_change_ratio: Option<DecimalEnclosure>,
    pub interpretation: String,
}
/// Signed adjacent-N changes and their exact preceding-change denominator.
/// Only N/dimension/input identity may vary; source precision, quadrature,
/// normalization and other design coordinates must agree. This reports finite
/// changes and does not turn two small increments into an infinite-tail bound.
pub fn stabilization_ladder(
    payloads: &[ObservationPayload],
    p: u32,
) -> Result<Vec<StabilizationStep>> {
    stabilization_ladder_checked(payloads, p, None)
}

/// An explicit quadrature schedule for a coupled N/order ladder. Exact realized
/// identities remain in each observation; the policy identifies the declared
/// rule generating this schedule and is retained with the result.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StabilizationQuadratureSchedule {
    pub policy_identity: String,
    pub identities_by_n_modes: BTreeMap<usize, String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScheduledStabilization {
    pub semantics: String,
    pub quadrature_schedule: StabilizationQuadratureSchedule,
    pub observation_digests: Vec<ConfigDigest>,
    pub steps: Vec<StabilizationStep>,
}

/// Compare a prespecified coupled N/quadrature ladder. This does not isolate
/// truncation error from quadrature error: both changes remain in the measured
/// increment. Other controls must still match exactly.
pub fn stabilization_ladder_with_quadrature_schedule(
    payloads: &[ObservationPayload],
    p: u32,
    schedule: &StabilizationQuadratureSchedule,
) -> Result<ScheduledStabilization> {
    if schedule.policy_identity.trim().is_empty()
        || schedule.identities_by_n_modes.len() != payloads.len()
        || schedule
            .identities_by_n_modes
            .values()
            .any(|s| s.trim().is_empty())
    {
        bail!("quadrature schedule must name a policy and exactly cover the ladder");
    }
    xc_core::validate_secret_free(schedule, "stabilization quadrature schedule")?;
    Ok(ScheduledStabilization {
        semantics: "finite-coupled-n-quadrature-stabilization-v1".into(),
        quadrature_schedule: schedule.clone(),
        observation_digests: payloads
            .iter()
            .map(xc_core::research_digest)
            .collect::<std::result::Result<_, _>>()?,
        steps: stabilization_ladder_checked(payloads, p, Some(schedule))?,
    })
}

fn stabilization_ladder_checked(
    payloads: &[ObservationPayload],
    p: u32,
    schedule: Option<&StabilizationQuadratureSchedule>,
) -> Result<Vec<StabilizationStep>> {
    precision(p)?;
    if payloads.len() < 2 {
        bail!("stabilization requires at least two observations");
    }
    let first = &payloads[0];
    first.validate()?;
    if matches!(
        first.observable.transform,
        ObservableTransform::SignedLogMagnitude { .. }
    ) {
        bail!("signed-log stabilization needs an explicit conversion");
    }
    let mut ranges = Vec::with_capacity(payloads.len());
    for x in payloads {
        x.validate()?;
        first.observable.check_compatible(&x.observable)?;
        let mut design = x.design.clone();
        design.n_modes = first.design.n_modes;
        design.dimension = first.design.dimension;
        design.exact_input_identity = first.design.exact_input_identity.clone();
        if let Some(schedule) = schedule {
            if x.design
                .n_modes
                .and_then(|n| schedule.identities_by_n_modes.get(&n))
                != Some(&x.design.quadrature_identity)
            {
                bail!("observation quadrature identity differs from the declared N schedule");
            }
            design.quadrature_identity = first.design.quadrature_identity.clone();
        }
        if design != first.design || x.design.n_modes.is_none() {
            bail!("stabilization mixes controls besides N/dimension/input identity");
        }
        ranges.push(resolution_range(&x.value, &x.resolution, p)?.0);
    }
    let mut previous_change: Option<MpfrInterval> = None;
    let mut result = vec![];
    for i in 1..payloads.len() {
        let from = payloads[i - 1].design.n_modes.unwrap();
        let to = payloads[i].design.n_modes.unwrap();
        if to <= from || payloads[i].design.dimension <= payloads[i - 1].design.dimension {
            bail!("stabilization ladder must increase N and dimension strictly");
        }
        let change = ranges[i]
            .as_ref()
            .zip(ranges[i - 1].as_ref())
            .map(|(b, a)| b.sub(a));
        let ratio = change
            .as_ref()
            .zip(previous_change.as_ref())
            .and_then(|(b, a)| {
                if a.contains_zero() {
                    None
                } else {
                    b.div(a).ok()
                }
            });
        result.push(StabilizationStep {
            from_n_modes: from, to_n_modes: to,
            signed_change: change.as_ref().map(DecimalEnclosure::from_interval),
            ratio_denominator: previous_change.as_ref().map(DecimalEnclosure::from_interval),
            signed_change_ratio: ratio.as_ref().map(DecimalEnclosure::from_interval),
            interpretation: if schedule.is_some() {
                "Finite coupled N/quadrature change; neither a pure truncation error nor an infinite-N remainder bound. Shared-input dependencies are enclosed conservatively; a zero-containing denominator leaves the ratio unresolved."
            } else {
                "Finite adjacent change, not an infinite-N remainder bound. Shared-input dependencies are enclosed conservatively, not assumed independent; a zero-containing denominator leaves the ratio unresolved."
            }.into(),
        });
        previous_change = change;
    }
    Ok(result)
}

/// Exact selected observation bytes, including unavailable reads. Payloads are
/// read only behind the scorer's cohort/protected-partition gates.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationObservation {
    pub metadata_digest: ConfigDigest,
    pub payload: Option<String>,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HypothesisEvaluationPacket {
    pub schema_version: u32,
    pub semantics: String,
    pub frozen: FrozenHypothesis,
    pub inventory: Vec<ObservationMetadata>,
    pub selected_observations: Vec<EvaluationObservation>,
    pub score: HypothesisScore,
    pub producer_toolkit_version: String,
}

/// Evaluate and retain an independently replayable scalar packet. Unselected
/// payloads are never loaded. Secret-bearing/non-UTF8 responses are treated as
/// unavailable and are not embedded. Matrix/root bytes are separate dependencies.
pub fn evaluate_hypothesis_packet<F>(
    frozen: &FrozenHypothesis,
    metadata: &[ObservationMetadata],
    partition: DatasetPartition,
    allow_protected_validation: bool,
    p: u32,
    mut load: F,
) -> Result<HypothesisEvaluationPacket>
where
    F: FnMut(&ObservationMetadata) -> Result<Vec<u8>>,
{
    let mut selected = Vec::new();
    let result = score_frozen_hypothesis(
        frozen,
        metadata,
        partition,
        allow_protected_validation,
        p,
        |m| {
            let payload = load(m)
                .ok()
                .and_then(|bytes| String::from_utf8(bytes).ok())
                .filter(|text| xc_core::validate_secret_free(text, "observation bytes").is_ok());
            selected.push(EvaluationObservation {
                metadata_digest: xc_core::research_digest(m)?,
                payload: payload.clone(),
            });
            payload
                .map(String::into_bytes)
                .ok_or_else(|| anyhow::anyhow!("retained observation unavailable"))
        },
    )?;
    let packet = HypothesisEvaluationPacket {
        schema_version: 1,
        semantics: "replayable-hypothesis-evaluation-v1".into(),
        frozen: frozen.clone(),
        inventory: metadata.to_vec(),
        selected_observations: selected,
        score: result
            .value
            .ok_or_else(|| anyhow::anyhow!("scorer did not return a report"))?,
        producer_toolkit_version: env!("CARGO_PKG_VERSION").into(),
    };
    xc_core::validate_secret_free(&packet, "hypothesis evaluation packet")?;
    Ok(packet)
}
impl HypothesisEvaluationPacket {
    /// Replay every selected scalar and verdict from embedded bytes. This does
    /// not authenticate or recompute the referenced matrix/root source payloads.
    pub fn validate(&self, allow_protected_validation: bool) -> Result<()> {
        if self.schema_version != 1 || self.semantics != "replayable-hypothesis-evaluation-v1" {
            bail!("unsupported hypothesis evaluation packet");
        }
        xc_cache::ToolkitVersion::parse(&self.producer_toolkit_version)?;
        xc_core::validate_secret_free(self, "hypothesis evaluation packet")?;
        let mut visited = BTreeSet::new();
        let mut reads = BTreeMap::new();
        for r in &self.selected_observations {
            if !r.metadata_digest.is_sha256() || reads.insert(&r.metadata_digest.0, r).is_some() {
                bail!("duplicate or invalid evaluation read identity");
            }
        }
        let replay = score_frozen_hypothesis(
            &self.frozen,
            &self.inventory,
            self.score.partition,
            allow_protected_validation,
            self.score.analysis_precision_bits,
            |m| {
                let digest = xc_core::research_digest(m)?;
                visited.insert(digest.0.clone());
                let r = reads
                    .get(&digest.0)
                    .ok_or_else(|| anyhow::anyhow!("evaluation read missing"))?;
                r.payload
                    .as_ref()
                    .map(|s| s.as_bytes().to_vec())
                    .ok_or_else(|| anyhow::anyhow!("retained observation unavailable"))
            },
        )?;
        if visited != reads.keys().map(|s| (*s).clone()).collect()
            || replay.value.as_ref() != Some(&self.score)
        {
            bail!("evaluation packet read coverage or score replay mismatch");
        }
        Ok(())
    }
}

/// Persist a replay-checked evaluation with the complete exact dependency set
/// of successfully loaded observations. Full hypotheses are private-only.
pub fn persist_hypothesis_evaluation(
    packet: &HypothesisEvaluationPacket,
    sources: &[xc_cache::ArtifactManifest],
    allow_protected_validation: bool,
    cache: &xc_cache::ArtifactCacheContext<'_>,
) -> Result<xc_cache::ArtifactExecutionCacheResult<HypothesisEvaluationPacket>> {
    packet.validate(allow_protected_validation)?;
    let dependencies = xc_cache::research_source_dependencies(sources)?;
    let mut expected = BTreeSet::new();
    for c in &packet.score.cases {
        if let Some(digest) = &c.source_payload {
            let m = packet
                .inventory
                .iter()
                .find(|m| m.case_id == c.case_id && &m.payload_sha256 == digest)
                .ok_or_else(|| anyhow::anyhow!("scored observation metadata missing"))?;
            for source in &m.sources {
                expected.insert((
                    source.kind.clone(),
                    source.logical_key.clone(),
                    source.semantic_digest.clone(),
                    source.payload_digest.clone(),
                ));
            }
        }
    }
    let actual = dependencies
        .iter()
        .map(|d| {
            (
                d.key.kind.clone(),
                d.key.logical_key.clone(),
                d.key.parameters_digest.0.clone(),
                d.content_digest.0.clone(),
            )
        })
        .collect::<BTreeSet<_>>();
    if actual != expected {
        bail!("evaluation source manifests do not match the exact scored observation dependencies");
    }
    Ok(xc_cache::persist_research_artifact(
        xc_cache::HYPOTHESIS_EVALUATION_KIND,
        packet,
        &dependencies,
        cache,
        |value| {
            value
                .validate(allow_protected_validation)
                .map_err(|e| xc_cache::CacheError::InvalidManifest(e.to_string()))
        },
    )?)
}
