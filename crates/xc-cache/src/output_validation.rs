//! Byte-for-byte output-preservation reports for cache verification runs.

use crate::{
    ArtifactKey, ArtifactManifest, CacheError, ContentDigest, DependencyRef, SemanticKeyEnvelope,
    ToolkitVersion,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactOutputComparisonStatus {
    Match,
    Mismatch,
    ReferenceAbsent,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactDivergenceOrigin {
    NotDiverged,
    FirstDivergence,
    InheritedFromDependency,
    Undetermined,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputValidationRunStatus {
    InProgress,
    Completed,
    Aborted,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactOutputComparison {
    pub schema_version: u32,
    pub operation: String,
    pub artifact_key: ArtifactKey,
    pub semantic_key: SemanticKeyEnvelope,
    pub computed_manifest_digest: ContentDigest,
    pub computed_provenance_digest: Option<ContentDigest>,
    pub computed_payload_digest: ContentDigest,
    pub computed_payload_size_bytes: u64,
    pub computed_dependencies: Vec<DependencyRef>,
    pub reference_overlay: Option<String>,
    pub reference_manifest_digest: Option<ContentDigest>,
    pub reference_provenance_digest: Option<ContentDigest>,
    pub reference_payload_digest: Option<ContentDigest>,
    pub reference_payload_size_bytes: Option<u64>,
    pub reference_producer_toolkit_version: Option<ToolkitVersion>,
    pub reference_absence_reason: Option<String>,
    pub status: ArtifactOutputComparisonStatus,
    pub divergence_origin: ArtifactDivergenceOrigin,
    pub diverging_dependencies: Vec<ArtifactKey>,
    pub first_differing_byte_offset: Option<u64>,
    pub intra_run_nondeterminism: bool,
    pub seed_source: Option<DependencyRef>,
    pub compute_duration_millis: u64,
    pub reference_fetch_duration_millis: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactOutputComparisonTotals {
    pub compared: usize,
    pub matched: usize,
    pub mismatched: usize,
    pub reference_absent: usize,
    pub first_divergences: usize,
    pub inherited_divergences: usize,
    pub undetermined_divergences: usize,
    pub intra_run_nondeterminism: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputPreservationValidationReport {
    pub schema_version: u32,
    pub run_id: ContentDigest,
    pub started_unix_seconds: u64,
    pub completed_unix_seconds: u64,
    pub toolkit_version: ToolkitVersion,
    pub candidate_source_revision: Option<String>,
    pub reference_mode: String,
    pub ordered_reference_overlays: Vec<String>,
    pub validation_root: String,
    pub production_cache_installed: bool,
    pub remote_publication_enabled: bool,
    pub run_status: OutputValidationRunStatus,
    pub comparisons: Vec<ArtifactOutputComparison>,
    pub totals: ArtifactOutputComparisonTotals,
    pub output_preserving: bool,
}

impl OutputPreservationValidationReport {
    pub fn validate(&self) -> Result<(), CacheError> {
        if self.schema_version != 1
            || !self.run_id.validate()
            || self.reference_mode.trim().is_empty()
            || self.validation_root.trim().is_empty()
            || self.ordered_reference_overlays.is_empty()
            || self.production_cache_installed
            || self.remote_publication_enabled
            || self
                .candidate_source_revision
                .as_ref()
                .is_some_and(|revision| {
                    revision.len() != 40
                        || !revision
                            .bytes()
                            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
                })
        {
            return Err(CacheError::InvalidManifest(
                "output-validation report metadata is invalid".to_owned(),
            ));
        }
        let totals = comparison_totals(&self.comparisons);
        if totals != self.totals {
            return Err(CacheError::InvalidManifest(
                "output-validation report totals disagree with its comparisons".to_owned(),
            ));
        }
        let expected = self.run_status == OutputValidationRunStatus::Completed
            && !self.comparisons.is_empty()
            && self
                .comparisons
                .iter()
                .all(|item| item.status == ArtifactOutputComparisonStatus::Match)
            && self.totals.intra_run_nondeterminism == 0;
        if self.output_preserving != expected {
            return Err(CacheError::InvalidManifest(
                "output-validation pass flag disagrees with its comparisons".to_owned(),
            ));
        }
        for pair in self.comparisons.windows(2) {
            if comparison_sort_key(&pair[0]) >= comparison_sort_key(&pair[1]) {
                return Err(CacheError::InvalidManifest(
                    "output-validation comparisons are not uniquely ordered".to_owned(),
                ));
            }
        }
        for comparison in &self.comparisons {
            validate_comparison(comparison)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OutputValidationRunConfig {
    pub validation_root: PathBuf,
    pub report_root: PathBuf,
    pub toolkit_version: ToolkitVersion,
    pub reference_mode: String,
    pub ordered_reference_overlays: Vec<String>,
    pub production_cache_installed: bool,
    pub remote_publication_enabled: bool,
}

#[derive(Clone, Debug)]
struct ActiveOutputValidationRun {
    config: OutputValidationRunConfig,
    started_unix_seconds: u64,
    records: BTreeMap<String, ArtifactOutputComparison>,
}

fn claim_scope_active() -> &'static Mutex<bool> {
    static ACTIVE: OnceLock<Mutex<bool>> = OnceLock::new();
    ACTIVE.get_or_init(|| Mutex::new(false))
}

/// Keeps sequential managed cache sessions inside one output-validation claim.
/// Session finalizers write cumulative checkpoints while this scope is active;
/// [`OutputValidationClaim::finish`] writes the sole terminal verdict.
#[must_use = "an output-validation claim must be finished to emit its terminal verdict"]
pub struct OutputValidationClaim {
    enabled: bool,
    finished: bool,
}

impl OutputValidationClaim {
    pub fn from_environment() -> Result<Self, CacheError> {
        let enabled = matches!(
            std::env::var("XC_CACHE_MODE").ok().as_deref(),
            Some("verify" | "verify_against_reference" | "verify-against-reference")
        );
        Self::begin(enabled)
    }

    pub fn begin(enabled: bool) -> Result<Self, CacheError> {
        if enabled {
            let mut active = claim_scope_active().lock().map_err(|_| {
                CacheError::InvalidTransition(
                    "output-validation claim-scope lock was poisoned".to_owned(),
                )
            })?;
            if *active {
                return Err(CacheError::InvalidTransition(
                    "an output-validation claim scope is already active in this process".to_owned(),
                ));
            }
            *active = true;
        }
        Ok(Self {
            enabled,
            finished: false,
        })
    }

    pub fn finish(mut self, claim_succeeded: bool) -> Result<Option<PathBuf>, CacheError> {
        if !self.enabled {
            self.finished = true;
            return Ok(None);
        }
        set_claim_scope_inactive()?;
        if !claim_succeeded
            && !active_run()
                .lock()
                .map_err(|_| {
                    CacheError::InvalidTransition(
                        "output-validation run lock was poisoned".to_owned(),
                    )
                })?
                .is_some()
        {
            self.finished = true;
            return Ok(None);
        }
        let result = finish_output_validation_run(claim_succeeded).map(Some);
        self.finished = true;
        result
    }
}

#[cfg(test)]
pub(crate) fn output_validation_test_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

impl Drop for OutputValidationClaim {
    fn drop(&mut self) {
        if self.enabled && !self.finished {
            if let Ok(mut active) = claim_scope_active().lock() {
                *active = false;
            }
            let has_active_run = active_run()
                .lock()
                .map(|active| active.is_some())
                .unwrap_or(false);
            if has_active_run {
                if let Err(error) = finish_output_validation_run(false) {
                    eprintln!(
                        "output validation: failed to persist aborted claim during cleanup: {error}"
                    );
                }
            }
        }
    }
}

fn set_claim_scope_inactive() -> Result<(), CacheError> {
    let mut active = claim_scope_active().lock().map_err(|_| {
        CacheError::InvalidTransition("output-validation claim-scope lock was poisoned".to_owned())
    })?;
    *active = false;
    Ok(())
}

pub(crate) fn output_validation_claim_scope_is_active() -> Result<bool, CacheError> {
    claim_scope_active()
        .lock()
        .map(|active| *active)
        .map_err(|_| {
            CacheError::InvalidTransition(
                "output-validation claim-scope lock was poisoned".to_owned(),
            )
        })
}

fn active_run() -> &'static Mutex<Option<ActiveOutputValidationRun>> {
    static ACTIVE: OnceLock<Mutex<Option<ActiveOutputValidationRun>>> = OnceLock::new();
    ACTIVE.get_or_init(|| Mutex::new(None))
}

pub(crate) fn begin_output_validation_run(
    config: OutputValidationRunConfig,
) -> Result<(), CacheError> {
    if config.validation_root.as_os_str().is_empty()
        || config.report_root.as_os_str().is_empty()
        || config.reference_mode.trim().is_empty()
        || config.ordered_reference_overlays.is_empty()
        || config.production_cache_installed
        || config.remote_publication_enabled
    {
        return Err(CacheError::InvalidManifest(
            "output-validation run configuration is incomplete".to_owned(),
        ));
    }
    let mut active = active_run().lock().map_err(|_| {
        CacheError::InvalidTransition("output-validation run lock was poisoned".to_owned())
    })?;
    if let Some(run) = active.as_ref() {
        if output_validation_claim_scope_is_active()? && run.config == config {
            return Ok(());
        }
        return Err(CacheError::InvalidTransition(
            "an output-validation run is already active in this process".to_owned(),
        ));
    }
    *active = Some(ActiveOutputValidationRun {
        config,
        started_unix_seconds: unix_seconds()?,
        records: BTreeMap::new(),
    });
    Ok(())
}

pub(crate) struct OutputComparisonInput<'a> {
    pub operation: &'a str,
    pub semantic_key: &'a SemanticKeyEnvelope,
    pub computed_manifest: &'a ArtifactManifest,
    pub computed_payload: &'a [u8],
    pub computed_dependencies: &'a [DependencyRef],
    pub reference: Option<(&'a str, &'a ArtifactManifest, &'a [u8])>,
    pub reference_absence_reason: Option<String>,
    pub seed_source: Option<DependencyRef>,
    pub compute_duration_millis: u64,
    pub reference_fetch_duration_millis: u64,
}

pub(crate) fn record_output_comparison(
    input: OutputComparisonInput<'_>,
) -> Result<ArtifactOutputComparisonStatus, CacheError> {
    let computed_payload_digest = ContentDigest::sha256(input.computed_payload);
    if computed_payload_digest != input.computed_manifest.content_digest {
        return Err(CacheError::InvalidManifest(
            "computed validation manifest does not match its payload".to_owned(),
        ));
    }
    let (
        status,
        overlay,
        reference_manifest_digest,
        reference_provenance_digest,
        payload_digest,
        payload_size,
        producer,
        first_offset,
    ) = match input.reference {
        Some((overlay, manifest, payload)) => {
            let reference_digest = ContentDigest::sha256(payload);
            if reference_digest != manifest.content_digest {
                return Err(CacheError::DigestMismatch {
                    expected: manifest.content_digest.0.clone(),
                    actual: reference_digest.0,
                });
            }
            let status = if payload == input.computed_payload {
                ArtifactOutputComparisonStatus::Match
            } else {
                ArtifactOutputComparisonStatus::Mismatch
            };
            (
                status,
                Some(overlay.to_owned()),
                Some(manifest_digest(manifest)?),
                manifest.provenance_digest.clone(),
                Some(manifest.content_digest.clone()),
                Some(manifest.size_bytes),
                Some(manifest.producer_toolkit_version.clone()),
                first_differing_byte(input.computed_payload, payload),
            )
        }
        None => (
            ArtifactOutputComparisonStatus::ReferenceAbsent,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
    };
    let mut comparison = ArtifactOutputComparison {
        schema_version: 1,
        operation: input.operation.to_owned(),
        artifact_key: input.computed_manifest.key.clone(),
        semantic_key: input.semantic_key.clone(),
        computed_manifest_digest: manifest_digest(input.computed_manifest)?,
        computed_provenance_digest: input.computed_manifest.provenance_digest.clone(),
        computed_payload_digest,
        computed_payload_size_bytes: input.computed_payload.len() as u64,
        computed_dependencies: input.computed_dependencies.to_vec(),
        reference_overlay: overlay,
        reference_manifest_digest,
        reference_provenance_digest,
        reference_payload_digest: payload_digest,
        reference_payload_size_bytes: payload_size,
        reference_producer_toolkit_version: producer,
        reference_absence_reason: input.reference_absence_reason,
        status,
        divergence_origin: if status == ArtifactOutputComparisonStatus::Match {
            ArtifactDivergenceOrigin::NotDiverged
        } else {
            ArtifactDivergenceOrigin::FirstDivergence
        },
        diverging_dependencies: Vec::new(),
        first_differing_byte_offset: first_offset,
        intra_run_nondeterminism: false,
        seed_source: input.seed_source,
        compute_duration_millis: input.compute_duration_millis,
        reference_fetch_duration_millis: input.reference_fetch_duration_millis,
    };
    validate_comparison(&comparison)?;
    let identity = artifact_identity_key(&comparison.artifact_key);
    let mut active = active_run().lock().map_err(|_| {
        CacheError::InvalidTransition("output-validation run lock was poisoned".to_owned())
    })?;
    let run = active.as_mut().ok_or_else(|| {
        CacheError::InvalidTransition(
            "output comparison was recorded without an active validation run".to_owned(),
        )
    })?;
    if let Some(existing) = run.records.get_mut(&identity) {
        if existing.computed_payload_digest != comparison.computed_payload_digest {
            existing.intra_run_nondeterminism = true;
        }
    } else {
        comparison.intra_run_nondeterminism = false;
        run.records.insert(identity, comparison);
    }
    Ok(status)
}

pub(crate) fn finalize_output_validation_run() -> Result<PathBuf, CacheError> {
    finish_output_validation_run(true)
}

pub(crate) fn checkpoint_output_validation_run() -> Result<PathBuf, CacheError> {
    let active = active_run().lock().map_err(|_| {
        CacheError::InvalidTransition("output-validation run lock was poisoned".to_owned())
    })?;
    let run = active.as_ref().cloned().ok_or_else(|| {
        CacheError::InvalidTransition("no output-validation run is active".to_owned())
    })?;
    drop(active);
    persist_output_validation_report(run, OutputValidationRunStatus::InProgress, false)
}

fn finish_output_validation_run(claim_succeeded: bool) -> Result<PathBuf, CacheError> {
    let mut active = active_run().lock().map_err(|_| {
        CacheError::InvalidTransition("output-validation run lock was poisoned".to_owned())
    })?;
    let run = active.take().ok_or_else(|| {
        CacheError::InvalidTransition("no output-validation run is active".to_owned())
    })?;
    drop(active);

    let status = if claim_succeeded {
        OutputValidationRunStatus::Completed
    } else {
        OutputValidationRunStatus::Aborted
    };
    persist_output_validation_report(run, status, claim_succeeded)
}

fn persist_output_validation_report(
    run: ActiveOutputValidationRun,
    run_status: OutputValidationRunStatus,
    enforce_output_preserving: bool,
) -> Result<PathBuf, CacheError> {
    let mut comparisons = run.records.into_values().collect::<Vec<_>>();
    classify_divergences(&mut comparisons);
    comparisons.sort_by_key(comparison_sort_key);
    let totals = comparison_totals(&comparisons);
    let output_preserving = run_status == OutputValidationRunStatus::Completed
        && !comparisons.is_empty()
        && totals.mismatched == 0
        && totals.reference_absent == 0
        && totals.intra_run_nondeterminism == 0;
    let completed_unix_seconds = unix_seconds()?;
    let report_root = run.config.report_root.clone();
    let provisional = OutputPreservationValidationReport {
        schema_version: 1,
        run_id: ContentDigest("0".repeat(64)),
        started_unix_seconds: run.started_unix_seconds,
        completed_unix_seconds,
        toolkit_version: run.config.toolkit_version,
        candidate_source_revision: option_env!("XC_SOURCE_REVISION").map(str::to_owned),
        reference_mode: run.config.reference_mode,
        ordered_reference_overlays: run.config.ordered_reference_overlays,
        validation_root: run.config.validation_root.display().to_string(),
        production_cache_installed: run.config.production_cache_installed,
        remote_publication_enabled: run.config.remote_publication_enabled,
        run_status,
        comparisons,
        totals,
        output_preserving,
    };
    let mut report = provisional;
    report.run_id = ContentDigest::sha256(&serde_json::to_vec(&report)?);
    report.validate()?;

    fs::create_dir_all(&report_root)?;
    let path = report_root.join(format!(
        "{:020}-{}.json",
        report.started_unix_seconds, report.run_id.0
    ));
    let bytes = serde_json::to_vec_pretty(&report)?;
    crate::atomic_replace(&path, &bytes)?;
    crate::atomic_replace(&report_root.join("latest.json"), &bytes)?;
    eprintln!(
        "output validation: status={:?}, compared={}, matched={}, mismatched={}, absent={}, first_divergences={}, report={}",
        report.run_status,
        report.totals.compared,
        report.totals.matched,
        report.totals.mismatched,
        report.totals.reference_absent,
        report.totals.first_divergences,
        path.display()
    );
    if enforce_output_preserving && !report.output_preserving {
        return Err(CacheError::InvalidTransition(format!(
            "output-preservation validation failed: compared={}, mismatched={}, reference_absent={}, nondeterministic={}; report: {}",
            report.totals.compared,
            report.totals.mismatched,
            report.totals.reference_absent,
            report.totals.intra_run_nondeterminism,
            path.display()
        )));
    }
    Ok(path)
}

fn validate_comparison(comparison: &ArtifactOutputComparison) -> Result<(), CacheError> {
    comparison.semantic_key.validate()?;
    if comparison.schema_version != 1
        || comparison.operation.trim().is_empty()
        || !comparison.computed_manifest_digest.validate()
        || !comparison.computed_payload_digest.validate()
        || comparison
            .computed_provenance_digest
            .as_ref()
            .is_some_and(|digest| !digest.validate())
        || comparison
            .reference_provenance_digest
            .as_ref()
            .is_some_and(|digest| !digest.validate())
        || comparison.artifact_key.kind != comparison.semantic_key.artifact_kind
        || comparison.artifact_key.parameters_digest != comparison.semantic_key.digest()?
    {
        return Err(CacheError::InvalidManifest(
            "output comparison identity is invalid".to_owned(),
        ));
    }
    for dependency in comparison
        .computed_dependencies
        .iter()
        .chain(comparison.seed_source.iter())
    {
        if !dependency.key.parameters_digest.validate() || !dependency.content_digest.validate() {
            return Err(CacheError::InvalidManifest(
                "output comparison contains an invalid dependency identity".to_owned(),
            ));
        }
    }
    if comparison
        .diverging_dependencies
        .iter()
        .any(|dependency| !dependency.parameters_digest.validate())
    {
        return Err(CacheError::InvalidManifest(
            "output comparison contains an invalid diverging dependency".to_owned(),
        ));
    }
    match comparison.status {
        ArtifactOutputComparisonStatus::Match => {
            if comparison.reference_manifest_digest.is_none()
                || comparison.reference_overlay.is_none()
                || comparison.reference_producer_toolkit_version.is_none()
                || comparison.reference_payload_digest.as_ref()
                    != Some(&comparison.computed_payload_digest)
                || comparison.reference_payload_size_bytes
                    != Some(comparison.computed_payload_size_bytes)
                || comparison.first_differing_byte_offset.is_some()
            {
                return Err(CacheError::InvalidManifest(
                    "matching output comparison contains unequal payload evidence".to_owned(),
                ));
            }
        }
        ArtifactOutputComparisonStatus::Mismatch => {
            if comparison.reference_payload_digest.is_none()
                || comparison.reference_manifest_digest.is_none()
                || comparison.first_differing_byte_offset.is_none()
            {
                return Err(CacheError::InvalidManifest(
                    "mismatching output comparison lacks reference evidence".to_owned(),
                ));
            }
        }
        ArtifactOutputComparisonStatus::ReferenceAbsent => {
            if comparison.reference_payload_digest.is_some()
                || comparison.reference_manifest_digest.is_some()
                || comparison.reference_provenance_digest.is_some()
                || comparison.reference_overlay.is_some()
                || comparison.reference_producer_toolkit_version.is_some()
                || comparison.reference_payload_size_bytes.is_some()
                || comparison.reference_absence_reason.is_none()
            {
                return Err(CacheError::InvalidManifest(
                    "absent-reference comparison contains contradictory evidence".to_owned(),
                ));
            }
        }
    }
    Ok(())
}

fn classify_divergences(comparisons: &mut [ArtifactOutputComparison]) {
    let statuses = comparisons
        .iter()
        .map(|item| (artifact_identity_key(&item.artifact_key), item.status))
        .collect::<BTreeMap<_, _>>();
    for item in comparisons {
        if item.status == ArtifactOutputComparisonStatus::Match {
            item.divergence_origin = ArtifactDivergenceOrigin::NotDiverged;
            item.diverging_dependencies.clear();
            continue;
        }
        let mut missing = false;
        let mut diverging = Vec::new();
        for dependency in &item.computed_dependencies {
            match statuses.get(&artifact_identity_key(&dependency.key)) {
                Some(ArtifactOutputComparisonStatus::Match) => {}
                Some(_) => diverging.push(dependency.key.clone()),
                None => missing = true,
            }
        }
        if let Some(seed) = &item.seed_source {
            if !item
                .computed_dependencies
                .iter()
                .any(|dependency| dependency.key == seed.key)
            {
                match statuses.get(&artifact_identity_key(&seed.key)) {
                    Some(ArtifactOutputComparisonStatus::Match) => {}
                    Some(_) => diverging.push(seed.key.clone()),
                    None => missing = true,
                }
            }
        }
        item.diverging_dependencies = diverging;
        item.divergence_origin = if !item.diverging_dependencies.is_empty() {
            ArtifactDivergenceOrigin::InheritedFromDependency
        } else if missing {
            ArtifactDivergenceOrigin::Undetermined
        } else {
            ArtifactDivergenceOrigin::FirstDivergence
        };
    }
}

fn comparison_totals(comparisons: &[ArtifactOutputComparison]) -> ArtifactOutputComparisonTotals {
    let mut totals = ArtifactOutputComparisonTotals {
        compared: comparisons.len(),
        ..ArtifactOutputComparisonTotals::default()
    };
    for item in comparisons {
        match item.status {
            ArtifactOutputComparisonStatus::Match => totals.matched += 1,
            ArtifactOutputComparisonStatus::Mismatch => totals.mismatched += 1,
            ArtifactOutputComparisonStatus::ReferenceAbsent => totals.reference_absent += 1,
        }
        match item.divergence_origin {
            ArtifactDivergenceOrigin::NotDiverged => {}
            ArtifactDivergenceOrigin::FirstDivergence => totals.first_divergences += 1,
            ArtifactDivergenceOrigin::InheritedFromDependency => totals.inherited_divergences += 1,
            ArtifactDivergenceOrigin::Undetermined => totals.undetermined_divergences += 1,
        }
        if item.intra_run_nondeterminism {
            totals.intra_run_nondeterminism += 1;
        }
    }
    totals
}

fn comparison_sort_key(item: &ArtifactOutputComparison) -> String {
    artifact_identity_key(&item.artifact_key)
}

fn artifact_identity_key(key: &ArtifactKey) -> String {
    format!(
        "{}\n{}\n{}",
        key.kind, key.logical_key, key.parameters_digest.0
    )
}

fn manifest_digest(manifest: &ArtifactManifest) -> Result<ContentDigest, CacheError> {
    Ok(ContentDigest::sha256(&serde_json::to_vec(manifest)?))
}

fn first_differing_byte(left: &[u8], right: &[u8]) -> Option<u64> {
    left.iter()
        .zip(right)
        .position(|(left, right)| left != right)
        .or_else(|| (left.len() != right.len()).then_some(left.len().min(right.len())))
        .map(|offset| offset as u64)
}

fn unix_seconds() -> Result<u64, CacheError> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .map_err(|error| CacheError::InvalidTransition(format!("system clock error: {error}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CacheQuality, CacheVisibility};
    use serde_json::json;

    fn semantic(kind: &str, order: u64) -> SemanticKeyEnvelope {
        SemanticKeyEnvelope {
            schema_version: 1,
            artifact_kind: kind.to_owned(),
            mathematical_semantics_version: "validation-test-v1".to_owned(),
            resolved_mathematical_parameters: json!({"order": order}),
            normalization: None,
            target: None,
            subspace: None,
            source_data_identities: BTreeMap::new(),
            algorithm_semantics: None,
        }
    }

    fn manifest(
        semantic: &SemanticKeyEnvelope,
        logical_key: &str,
        payload: &[u8],
        dependencies: Vec<DependencyRef>,
    ) -> ArtifactManifest {
        ArtifactManifest {
            schema_version: 1,
            key: ArtifactKey {
                kind: semantic.artifact_kind.clone(),
                logical_key: logical_key.to_owned(),
                parameters_digest: semantic.digest().unwrap(),
            },
            content_digest: ContentDigest::sha256(payload),
            size_bytes: payload.len() as u64,
            objects: Vec::new(),
            created_unix_seconds: 1,
            producer_toolkit_version: ToolkitVersion::parse("0.13.3").unwrap(),
            minimum_reader_version: ToolkitVersion::parse("0.13.0").unwrap(),
            maximum_reader_version: None,
            quality: CacheQuality::Validated,
            visibility: CacheVisibility::Local,
            immutable: true,
            dependencies,
            tags: BTreeMap::new(),
            provenance_digest: None,
        }
    }

    fn config(name: &str) -> OutputValidationRunConfig {
        let root = std::env::temp_dir().join(format!("{name}-{}", std::process::id()));
        OutputValidationRunConfig {
            validation_root: root.clone(),
            report_root: root.join("reports"),
            toolkit_version: ToolkitVersion::parse("0.13.3").unwrap(),
            reference_mode: "fixture".to_owned(),
            ordered_reference_overlays: vec!["reference".to_owned()],
            production_cache_installed: false,
            remote_publication_enabled: false,
        }
    }

    fn record(
        operation: &str,
        semantic: &SemanticKeyEnvelope,
        computed: &ArtifactManifest,
        computed_payload: &[u8],
        reference: &ArtifactManifest,
        reference_payload: &[u8],
    ) -> ArtifactOutputComparisonStatus {
        record_output_comparison(OutputComparisonInput {
            operation,
            semantic_key: semantic,
            computed_manifest: computed,
            computed_payload,
            computed_dependencies: &computed.dependencies,
            reference: Some(("reference", reference, reference_payload)),
            reference_absence_reason: None,
            seed_source: None,
            compute_duration_millis: 1,
            reference_fetch_duration_millis: 1,
        })
        .unwrap()
    }

    #[test]
    fn first_differing_byte_handles_content_and_length() {
        assert_eq!(first_differing_byte(b"abc", b"axc"), Some(1));
        assert_eq!(first_differing_byte(b"abc", b"abcx"), Some(3));
        assert_eq!(first_differing_byte(b"abc", b"abc"), None);
    }

    #[test]
    fn claim_scope_aggregates_sequential_session_checkpoints() {
        let _guard = output_validation_test_lock().lock().unwrap();
        let passing = config("output-validation-multi-session");
        let _ = fs::remove_dir_all(&passing.validation_root);
        let claim = OutputValidationClaim::begin(true).unwrap();

        begin_output_validation_run(passing.clone()).unwrap();
        let leaf_semantic = semantic("leaf", 1);
        let leaf = manifest(&leaf_semantic, "leaf/1", b"leaf", Vec::new());
        assert_eq!(
            record(
                "leaf.compute",
                &leaf_semantic,
                &leaf,
                b"leaf",
                &leaf,
                b"leaf"
            ),
            ArtifactOutputComparisonStatus::Match
        );
        checkpoint_output_validation_run().unwrap();
        let checkpoint: OutputPreservationValidationReport =
            serde_json::from_slice(&fs::read(passing.report_root.join("latest.json")).unwrap())
                .unwrap();
        assert_eq!(checkpoint.run_status, OutputValidationRunStatus::InProgress);
        assert_eq!(checkpoint.totals.compared, 1);
        assert!(!checkpoint.output_preserving);

        begin_output_validation_run(passing.clone()).unwrap();
        let dependency = DependencyRef {
            key: leaf.key.clone(),
            content_digest: leaf.content_digest.clone(),
            required_quality: leaf.quality,
        };
        let parent_semantic = semantic("parent", 2);
        let parent = manifest(&parent_semantic, "parent/2", b"parent", vec![dependency]);
        assert_eq!(
            record(
                "parent.compute",
                &parent_semantic,
                &parent,
                b"parent",
                &parent,
                b"parent"
            ),
            ArtifactOutputComparisonStatus::Match
        );
        checkpoint_output_validation_run().unwrap();
        claim.finish(true).unwrap();

        let completed: OutputPreservationValidationReport =
            serde_json::from_slice(&fs::read(passing.report_root.join("latest.json")).unwrap())
                .unwrap();
        assert_eq!(completed.run_status, OutputValidationRunStatus::Completed);
        assert_eq!(completed.totals.compared, 2);
        assert_eq!(completed.totals.matched, 2);
        assert!(completed.output_preserving);
        let _ = fs::remove_dir_all(passing.validation_root);

        let failing = config("output-validation-multi-session-mismatch");
        let _ = fs::remove_dir_all(&failing.validation_root);
        let claim = OutputValidationClaim::begin(true).unwrap();
        begin_output_validation_run(failing.clone()).unwrap();
        let mismatch_semantic = semantic("mismatch", 3);
        let mismatch_computed = manifest(&mismatch_semantic, "mismatch/3", b"new", Vec::new());
        let mismatch_reference = manifest(&mismatch_semantic, "mismatch/3", b"old", Vec::new());
        record(
            "mismatch.compute",
            &mismatch_semantic,
            &mismatch_computed,
            b"new",
            &mismatch_reference,
            b"old",
        );
        checkpoint_output_validation_run().unwrap();
        begin_output_validation_run(failing.clone()).unwrap();
        let later_semantic = semantic("later", 4);
        let later = manifest(&later_semantic, "later/4", b"later", Vec::new());
        record(
            "later.compute",
            &later_semantic,
            &later,
            b"later",
            &later,
            b"later",
        );
        checkpoint_output_validation_run().unwrap();
        assert!(claim.finish(true).is_err());
        let failed: OutputPreservationValidationReport =
            serde_json::from_slice(&fs::read(failing.report_root.join("latest.json")).unwrap())
                .unwrap();
        assert_eq!(failed.run_status, OutputValidationRunStatus::Completed);
        assert_eq!(failed.totals.compared, 2);
        assert_eq!(failed.totals.mismatched, 1);
        assert_eq!(failed.totals.matched, 1);

        let aborted = config("output-validation-aborted-claim");
        let _ = fs::remove_dir_all(&aborted.validation_root);
        let claim = OutputValidationClaim::begin(true).unwrap();
        begin_output_validation_run(aborted.clone()).unwrap();
        let partial_semantic = semantic("partial", 5);
        let partial = manifest(&partial_semantic, "partial/5", b"partial", Vec::new());
        record(
            "partial.compute",
            &partial_semantic,
            &partial,
            b"partial",
            &partial,
            b"partial",
        );
        claim.finish(false).unwrap();
        let aborted_report: OutputPreservationValidationReport =
            serde_json::from_slice(&fs::read(aborted.report_root.join("latest.json")).unwrap())
                .unwrap();
        assert_eq!(
            aborted_report.run_status,
            OutputValidationRunStatus::Aborted
        );
        assert!(!aborted_report.output_preserving);
        let _ = fs::remove_dir_all(failing.validation_root);
        let _ = fs::remove_dir_all(aborted.validation_root);
    }

    #[test]
    fn dependency_classification_and_nondeterminism_are_reported() {
        let _guard = output_validation_test_lock().lock().unwrap();
        let run_config = config("output-validation-classification");
        let _ = fs::remove_dir_all(&run_config.validation_root);
        begin_output_validation_run(run_config.clone()).unwrap();

        let leaf_semantic = semantic("leaf", 10);
        let leaf_computed = manifest(&leaf_semantic, "leaf/10", b"leaf-new", Vec::new());
        let mut leaf_reference = manifest(&leaf_semantic, "leaf/10", b"leaf-old", Vec::new());
        leaf_reference.provenance_digest = Some(ContentDigest("f".repeat(64)));
        record(
            "leaf.compute",
            &leaf_semantic,
            &leaf_computed,
            b"leaf-new",
            &leaf_reference,
            b"leaf-old",
        );

        let leaf_dependency = DependencyRef {
            key: leaf_computed.key.clone(),
            content_digest: leaf_computed.content_digest.clone(),
            required_quality: leaf_computed.quality,
        };
        let parent_semantic = semantic("parent", 11);
        let parent_computed = manifest(
            &parent_semantic,
            "parent/11",
            b"parent-new",
            vec![leaf_dependency],
        );
        let parent_reference = manifest(&parent_semantic, "parent/11", b"parent-old", Vec::new());
        record(
            "parent.compute",
            &parent_semantic,
            &parent_computed,
            b"parent-new",
            &parent_reference,
            b"parent-old",
        );

        let unknown_semantic = semantic("unknown", 12);
        let unknown_dependency_semantic = semantic("external-seed", 99);
        let unknown_dependency_manifest =
            manifest(&unknown_dependency_semantic, "seed/99", b"seed", Vec::new());
        let unknown_computed =
            manifest(&unknown_semantic, "unknown/12", b"unknown-new", Vec::new());
        let unknown_reference =
            manifest(&unknown_semantic, "unknown/12", b"unknown-old", Vec::new());
        let seed_source = DependencyRef {
            key: unknown_dependency_manifest.key,
            content_digest: unknown_dependency_manifest.content_digest,
            required_quality: unknown_dependency_manifest.quality,
        };
        record_output_comparison(OutputComparisonInput {
            operation: "unknown.compute",
            semantic_key: &unknown_semantic,
            computed_manifest: &unknown_computed,
            computed_payload: b"unknown-new",
            computed_dependencies: &unknown_computed.dependencies,
            reference: Some(("reference", &unknown_reference, b"unknown-old")),
            reference_absence_reason: None,
            seed_source: Some(seed_source.clone()),
            compute_duration_millis: 1,
            reference_fetch_duration_millis: 1,
        })
        .unwrap();

        let later_semantic = semantic("later", 13);
        let later = manifest(&later_semantic, "later/13", b"later", Vec::new());
        record(
            "later.compute",
            &later_semantic,
            &later,
            b"later",
            &later,
            b"later",
        );
        assert!(finalize_output_validation_run().is_err());

        let report: OutputPreservationValidationReport =
            serde_json::from_slice(&fs::read(run_config.report_root.join("latest.json")).unwrap())
                .unwrap();
        let by_key = report
            .comparisons
            .iter()
            .map(|item| (item.artifact_key.logical_key.as_str(), item))
            .collect::<BTreeMap<_, _>>();
        assert_eq!(
            by_key["leaf/10"].divergence_origin,
            ArtifactDivergenceOrigin::FirstDivergence
        );
        assert_eq!(
            by_key["leaf/10"].reference_provenance_digest,
            Some(ContentDigest("f".repeat(64)))
        );
        assert_ne!(
            by_key["leaf/10"].reference_manifest_digest,
            by_key["leaf/10"].reference_provenance_digest
        );
        assert_eq!(
            by_key["parent/11"].divergence_origin,
            ArtifactDivergenceOrigin::InheritedFromDependency
        );
        assert_eq!(
            by_key["unknown/12"].divergence_origin,
            ArtifactDivergenceOrigin::Undetermined
        );
        assert_eq!(by_key["unknown/12"].seed_source, Some(seed_source));
        assert_eq!(report.totals.compared, 4);
        assert_eq!(report.totals.matched, 1);

        let nondeterminism = config("output-validation-nondeterminism");
        let _ = fs::remove_dir_all(&nondeterminism.validation_root);
        begin_output_validation_run(nondeterminism.clone()).unwrap();
        let semantic = semantic("nondeterministic", 20);
        let first = manifest(&semantic, "nondeterministic/20", b"first", Vec::new());
        let second = manifest(&semantic, "nondeterministic/20", b"second", Vec::new());
        record(
            "first.compute",
            &semantic,
            &first,
            b"first",
            &first,
            b"first",
        );
        record(
            "second.compute",
            &semantic,
            &second,
            b"second",
            &first,
            b"first",
        );
        assert!(finalize_output_validation_run().is_err());
        let report: OutputPreservationValidationReport = serde_json::from_slice(
            &fs::read(nondeterminism.report_root.join("latest.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(report.totals.intra_run_nondeterminism, 1);
        let _ = fs::remove_dir_all(run_config.validation_root);
        let _ = fs::remove_dir_all(nondeterminism.validation_root);
    }
}
