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

pub struct CapturedDiagnostic {
    pub value: serde_json::Value,
    /// Exact manifests supplied by the diagnostic's authenticated cache route.
    pub sources: Vec<ArtifactManifest>,
}
impl CapturedDiagnostic {
    pub fn new<T: Serialize>(
        value: &T,
        sources: Vec<ArtifactManifest>,
    ) -> Result<Self, CacheError> {
        let value = serde_json::to_value(value).map_err(|e| invalid(e.to_string()))?;
        xc_core::validate_secret_free(&value, "capture measurement")
            .map_err(|e| invalid(e.to_string()))?;
        Ok(Self { value, sources })
    }
    pub fn from_cached<T: Serialize>(
        result: ArtifactExecutionCacheResult<T>,
    ) -> Result<Self, CacheError> {
        let sources = result
            .produced_manifest
            .or(result.reused_manifest)
            .into_iter()
            .collect();
        Self::new(&result.value, sources)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapturedMeasurement {
    pub value: serde_json::Value,
    pub source_dependencies: Vec<DependencyRef>,
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
        if m.quality.admissible_rank() < CacheQuality::Validated.admissible_rank() {
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
    let resolved_plan = serde_json::to_value(plan).map_err(|e| invalid(e.to_string()))?;
    xc_core::validate_secret_free(&resolved_plan, "capture plan")
        .map_err(|e| invalid(e.to_string()))?;
    let mut receipt = CaptureReceipt::new(&resolved_plan, requested.clone())
        .map_err(|e| invalid(e.to_string()))?;
    let mut measurements = BTreeMap::new();
    for id in &requested {
        let outcome = match execute(id) {
            Ok(diagnostic) => {
                let measurement = (|| {
                    xc_core::validate_secret_free(&diagnostic.value, "capture measurement")
                        .map_err(|e| invalid(e.to_string()))?;
                    let m = CapturedMeasurement {
                        value: diagnostic.value,
                        source_dependencies: research_source_dependencies(&diagnostic.sources)?,
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
    let record = CaptureArtifact {
        schema_version: 1,
        semantics: CAPTURE_RECORD_SEMANTICS.into(),
        resolved_plan,
        requested_diagnostics: requested,
        receipt,
        measurements,
        source_dependencies,
    };
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
        minimum_reader_version: ToolkitVersion::parse("0.15.0")?,
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
            if source.quality.admissible_rank() < dependency.required_quality.admissible_rank() {
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
            producer_toolkit_version: ToolkitVersion::parse("0.15.0").unwrap(),
            minimum_reader_version: ToolkitVersion::parse("0.13.0").unwrap(),
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
            if quality.admissible_rank() >= CacheQuality::Validated.admissible_rank() {
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
                ToolkitVersion::parse("0.15.0").unwrap()
            );
        }
    }
    #[test]
    fn managed_receipts_reuse_exact_attempts_and_preserve_changed_outcomes() {
        let root = std::env::temp_dir().join(format!(
            "xc-research-record-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
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
