//! Application adapter for failure-isolated research capture from one primary run.
//!
//! The caller owns the resolved plan, receipt, and run journal. No diagnostic
//! changes the primary parity, acquisition policy, or retained numerical state.
use super::*;
use std::sync::Mutex;
use xc_cache::{ArtifactProductionSink, CapturedDiagnostic, ProducedArtifactRecord};

/// A completed primary calculation and its exact retained sources.
pub struct RetainedCcmRun {
    params: CcmParams,
    cfg: HighPrecConfig,
    primary: HighPrecResult,
    source: RetainedCcmSource,
    eigenpair: Option<ProducedArtifactRecord>,
    retained_records: Vec<ProducedArtifactRecord>,
    research_inputs: Option<crate::ccm::retained_evidence::ResearchInputs>,
    extended_inputs: Option<crate::ccm::extended_research::ExternalResearchInputs>,
    extended_input_error: Option<String>,
    extended_inputs_loaded: bool,
    extended_options: Option<crate::ccm::extended_research::ExtensionOptions>,
    run_once_prepared: bool,
    prepared_reference_jets: bool,
    prepared_reference: Option<crate::ccm::research_completion::ReferencePreparation>,
    run_once_sources: Vec<ArtifactManifest>,
    uflow_capture: Option<CapturedDiagnostic>,
    sectors: Option<CcmSectorGapResolution>,
    sector_record: Option<CapturedDiagnostic>,
    sector_error: Option<String>,
    sector_options: Option<CcmSectorAnalysisOptions>,
    sector_certificate: Option<
        xc_cache::ArtifactExecutionCacheResult<
            crate::ccm::sector_gap_certificate::PortableCcmSectorGapCertificate,
        >,
    >,
    sector_certificate_error: Option<String>,
    sector_certificate_options:
        Option<crate::ccm::sector_gap_certificate::CcmSectorGapCertificationOptions>,
    /// Diagnostics computed ahead of their turn; see `start_lookahead`.
    lookahead: Option<crate::ccm::convergence_capture::lookahead::Lane>,
    lookahead_enabled: bool,
    lookahead_started: bool,
    lookahead_reached: std::collections::BTreeSet<String>,
    /// Diagnostics the caller declared it will request; only these may be
    /// computed ahead. `None` computes nothing ahead.
    lookahead_requests: Option<std::collections::BTreeSet<String>>,
}

/// Look-ahead diagnostics to compute: declared requests not yet reached, in
/// the receipt's (sorted) order. Nothing without a declaration.
fn lookahead_ids<'a>(
    requests: Option<&std::collections::BTreeSet<String>>,
    reached: &std::collections::BTreeSet<String>,
) -> Vec<&'a str> {
    use crate::ccm::convergence_capture::lookahead;
    let Some(requests) = requests else {
        return Vec::new();
    };
    let mut ids = lookahead::FINITE
        .iter()
        .chain(lookahead::EXTENDED)
        .copied()
        .filter(|id| requests.contains(*id) && !reached.contains(*id))
        .collect::<Vec<_>>();
    ids.sort_unstable();
    ids
}

// Observe only the requested logical payloads. Keep the configured publication
// sink, including its verified transport adoption, in the execution path.
struct RecordingSink<'a> {
    next: Option<&'a dyn ArtifactProductionSink>,
    kinds: &'a [&'a str],
    records: Mutex<Vec<ProducedArtifactRecord>>,
}
impl<'a> RecordingSink<'a> {
    fn new(cache: &'a ArtifactCacheContext<'_>, kinds: &'a [&'a str]) -> Self {
        Self {
            next: cache.production_sink,
            kinds,
            records: Mutex::new(Vec::new()),
        }
    }
    fn observe(&self, artifact: &ProducedArtifactRecord) -> Result<(), CacheError> {
        if self.kinds.contains(&artifact.manifest.key.kind.as_str()) {
            let mut records = self.records.lock().map_err(|_| {
                CacheError::InvalidManifest("capture observer lock poisoned".into())
            })?;
            if !records.iter().any(|r| {
                r.manifest.key == artifact.manifest.key
                    && r.manifest.content_digest == artifact.manifest.content_digest
            }) {
                records.push(artifact.clone());
            }
        }
        Ok(())
    }
    fn finish(self) -> Result<Vec<ProducedArtifactRecord>> {
        self.records
            .into_inner()
            .map_err(|_| anyhow::anyhow!("capture observer lock poisoned"))
    }
}
impl ArtifactProductionSink for RecordingSink<'_> {
    fn contains_dependency_identity(
        &self,
        identity: &xc_cache::PayloadDependencyIdentity,
    ) -> Result<bool, CacheError> {
        self.next.map_or(Ok(false), |next| {
            next.contains_dependency_identity(identity)
        })
    }
    fn retained_canonical_manifest(
        &self,
        identity: &xc_cache::PayloadDependencyIdentity,
    ) -> Result<Option<xc_cache::CanonicalArtifactManifest>, CacheError> {
        self.next
            .map_or(Ok(None), |next| next.retained_canonical_manifest(identity))
    }
    fn retained_canonical_manifest_for_artifact(
        &self,
        key: &ArtifactKey,
        digest: &ContentDigest,
        quality: CacheQuality,
    ) -> Result<Option<xc_cache::CanonicalArtifactManifest>, CacheError> {
        self.next.map_or(Ok(None), |next| {
            next.retained_canonical_manifest_for_artifact(key, digest, quality)
        })
    }
    fn canonical_closure_complete(
        &self,
        identity: &xc_cache::PayloadDependencyIdentity,
    ) -> Result<bool, CacheError> {
        self.next
            .map_or(Ok(false), |next| next.canonical_closure_complete(identity))
    }
    fn mark_canonical_closure_complete(
        &self,
        identity: &xc_cache::PayloadDependencyIdentity,
    ) -> Result<(), CacheError> {
        self.next.map_or(Ok(()), |next| {
            next.mark_canonical_closure_complete(identity)
        })
    }
    fn contains_artifact(
        &self,
        key: &ArtifactKey,
        digest: &ContentDigest,
    ) -> Result<bool, CacheError> {
        self.next
            .map_or(Ok(false), |next| next.contains_artifact(key, digest))
    }
    fn retained_assurance(
        &self,
        key: &ArtifactKey,
        digest: &ContentDigest,
    ) -> Result<Option<ArtifactProductionAssessment>, CacheError> {
        self.next
            .map_or(Ok(None), |next| next.retained_assurance(key, digest))
    }
    fn supports_encoded_records(&self) -> bool {
        self.next
            .is_some_and(|next| next.supports_encoded_records())
    }
    fn requires_produced_payload(&self, key: &ArtifactKey) -> bool {
        self.kinds.contains(&key.kind.as_str())
            || self
                .next
                .is_some_and(|next| next.requires_produced_payload(key))
    }
    fn record_encoded(
        &self,
        artifact: xc_cache::EncodedArtifactRecord,
        encoded: &xc_cache::VerifiedEncodedPayload,
        transport: Option<&xc_cache::VerifiedTransportParts>,
    ) -> Result<(), CacheError> {
        self.next
            .ok_or_else(|| {
                CacheError::InvalidTransition("encoded publication has no configured sink".into())
            })?
            .record_encoded(artifact, encoded, transport)
    }
    fn record_assurance_requirement(
        &self,
        requirement: xc_cache::ArtifactAssuranceRequirement,
    ) -> Result<(), CacheError> {
        if let Some(next) = self.next {
            next.record_assurance_requirement(requirement)?;
        }
        Ok(())
    }
    fn record_assurance(
        &self,
        attestation: xc_cache::ArtifactAssuranceAttestation,
    ) -> Result<(), CacheError> {
        if let Some(next) = self.next {
            next.record_assurance(attestation)?;
        }
        Ok(())
    }
    fn record_evidence(&self, kind: &str, payload: &[u8]) -> Result<ContentDigest, CacheError> {
        match self.next {
            Some(next) => next.record_evidence(kind, payload),
            None => Ok(ContentDigest::sha256(payload)),
        }
    }
    fn record(&self, artifact: ProducedArtifactRecord) -> Result<(), CacheError> {
        self.observe(&artifact)?;
        if let Some(next) = self.next {
            next.record(artifact)?;
        }
        Ok(())
    }
    fn record_with_verified_encoded(
        &self,
        artifact: ProducedArtifactRecord,
        encoded: &xc_cache::VerifiedEncodedPayload,
    ) -> Result<(), CacheError> {
        self.observe(&artifact)?;
        if let Some(next) = self.next {
            next.record_with_verified_encoded(artifact, encoded)?;
        }
        Ok(())
    }
    fn record_with_verified_transport(
        &self,
        artifact: ProducedArtifactRecord,
        encoded: &xc_cache::VerifiedEncodedPayload,
        transport: &xc_cache::VerifiedTransportParts,
    ) -> Result<(), CacheError> {
        self.observe(&artifact)?;
        if let Some(next) = self.next {
            next.record_with_verified_transport(artifact, encoded, transport)?;
        }
        Ok(())
    }
}

fn observing<'a>(
    cache: &ArtifactCacheContext<'a>,
    sink: &'a dyn ArtifactProductionSink,
) -> ArtifactCacheContext<'a> {
    ArtifactCacheContext {
        resolver: cache.resolver,
        reference_resolver: cache.reference_resolver,
        acceptance: cache.acceptance,
        ordered_overlays: cache.ordered_overlays.clone(),
        mode: cache.mode,
        write_on_miss: cache.write_on_miss,
        write_visibility: cache.write_visibility,
        requested_assurance: cache.requested_assurance,
        certification_failure_policy: cache.certification_failure_policy,
        production_sink: Some(sink),
    }
}

fn recorded(mut records: Vec<ProducedArtifactRecord>) -> Result<CapturedDiagnostic> {
    if records.is_empty() {
        bail!("diagnostic returned without a retained artifact");
    }
    records.sort_by(|a, b| {
        (
            &a.manifest.key.kind,
            &a.manifest.key.logical_key,
            &a.manifest.key.parameters_digest.0,
            &a.manifest.content_digest.0,
        )
            .cmp(&(
                &b.manifest.key.kind,
                &b.manifest.key.logical_key,
                &b.manifest.key.parameters_digest.0,
                &b.manifest.content_digest.0,
            ))
    });
    let values = records
        .iter()
        .map(|r| serde_json::from_slice::<serde_json::Value>(&r.payload))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    // The measurement is exactly these retained payloads, so the receipt
    // references them instead of embedding a second copy.
    let manifests = records.into_iter().map(|r| r.manifest).collect::<Vec<_>>();
    Ok(CapturedDiagnostic::by_reference(
        &values,
        manifests.clone(),
        true,
        manifests,
    )?)
}

impl RetainedCcmRun {
    /// Execute an explicitly indexed reference-seeded claim once.
    pub fn seeded(
        params: &CcmParams,
        cfg: &HighPrecConfig,
        first: usize,
        seeds: &[Float],
        dataset: &ReferenceZeroDatasetIdentity,
        cache: &ArtifactCacheContext<'_>,
    ) -> Result<Self> {
        validate_reference_seed_dataset(
            RootArtifactMode::ReferenceSeededRefinement,
            Some(dataset),
            first,
            seeds.len(),
        )?;
        Self::run(
            params,
            cfg,
            RootAcquisition::ReferenceSeeded {
                first_root_index: first,
                seeds,
                dataset,
            },
            cache,
        )
    }
    /// Execute an independently acquired claim once, preserving its exact domain.
    pub fn independent(
        params: &CcmParams,
        cfg: &HighPrecConfig,
        target: &ZeroTarget,
        options: IndependentRootDiscoveryOptions,
        cache: &ArtifactCacheContext<'_>,
    ) -> Result<Self> {
        Self::run(
            params,
            cfg,
            RootAcquisition::Independent { target, options },
            cache,
        )
    }
    fn run(
        params: &CcmParams,
        cfg: &HighPrecConfig,
        acquisition: RootAcquisition<'_>,
        cache: &ArtifactCacheContext<'_>,
    ) -> Result<Self> {
        let sink = RecordingSink::new(
            cache,
            &[
                "ccm_weil_eigenpair",
                "ccm_secular_source",
                "ccm_root_discovery_window",
                "ccm_root_refinement",
            ],
        );
        let observed = observing(cache, &sink);
        let (primary, source) = xc_numerics::hp_runtime::run_hp(|| {
            run_inner_retaining_source(
                params,
                cfg,
                acquisition,
                CcmCacheRoute::Fabric(&observed),
                None,
            )
        })?;
        let retained_records = sink.finish()?;
        let eigenpair = retained_records
            .iter()
            .find(|r| {
                source.eigenpair_manifest.as_ref().is_some_and(|m| {
                    m.key == r.manifest.key && m.content_digest == r.manifest.content_digest
                })
            })
            .cloned();
        if let (Some(eigenpair), Some(matrix)) = (&source.eigenpair_manifest, &source.tau_manifest)
        {
            let registration = crate::ccm::research_cohort::CohortRegistration {
                schema_version: 1,
                eigenpair: eigenpair.clone(),
                matrix: matrix.clone(),
                root: source.root_manifest.clone(),
                secular: source.secular_manifest.clone(),
                assembly_policy:
                    "ccm-weil-form-v0.15.1-v3; requested quadrature; corrected symmetric Tau".into(),
                quadrature_policy: format!(
                    "archimedean HP GL policy={:?}; active order parameter={}",
                    cfg.research_quadrature_policy,
                    cfg.quadrature_identity_points()
                ),
            };
            if let Err(e) = crate::ccm::research_cohort::register(&registration) {
                xc_core::progress_message!("research cohort registration unavailable: {e}");
            }
        }
        Ok(Self {
            params: params.clone(),
            cfg: cfg.clone(),
            primary,
            source,
            eigenpair,
            retained_records,
            research_inputs: None,
            extended_inputs: None,
            extended_input_error: None,
            extended_inputs_loaded: false,
            extended_options: None,
            run_once_prepared: false,
            prepared_reference_jets: false,
            prepared_reference: None,
            run_once_sources: Vec::new(),
            uflow_capture: None,
            sectors: None,
            sector_record: None,
            sector_error: None,
            sector_options: None,
            sector_certificate: None,
            sector_certificate_error: None,
            sector_certificate_options: None,
            lookahead: None,
            lookahead_enabled: true,
            lookahead_started: false,
            lookahead_reached: std::collections::BTreeSet::new(),
            lookahead_requests: None,
        })
    }
    /// Explicit additional references; the legacy runtime target file is never imported.
    pub fn set_research_inputs(
        &mut self,
        inputs: crate::ccm::retained_evidence::ResearchInputs,
    ) -> Result<()> {
        inputs.validate()?;
        self.research_inputs = Some(inputs);
        Ok(())
    }
    /// Load the non-executable research-input JSON from an explicitly selected file.
    pub fn load_research_inputs(&mut self, path: &std::path::Path) -> Result<()> {
        if std::fs::metadata(path)?.len() > 16 * 1024 * 1024 {
            bail!("research input file exceeds 16 MiB");
        }
        let bytes = std::fs::read(path)?;
        self.set_research_inputs(serde_json::from_slice(&bytes)?)
    }

    /// Configure bounded retained diagnostics; source precision and identity remain unchanged.
    pub fn set_extended_research_options(
        &mut self,
        options: crate::ccm::extended_research::ExtensionOptions,
    ) {
        self.extended_options = Some(options);
    }
    /// Explicit data-only source. No target formula or program is executed.
    pub fn set_extended_research_inputs(
        &mut self,
        inputs: crate::ccm::extended_research::ExternalResearchInputs,
    ) -> Result<()> {
        let record = self
            .eigenpair
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("retained primary state unavailable"))?;
        let state = crate::ccm::state_geometry::RetainedState::from_payload(
            &record.manifest,
            &record.payload,
            std::slice::from_ref(&record.manifest.content_digest),
        )?;
        inputs.matches(&state)?;
        self.run_once_prepared = false;
        self.run_once_sources.clear();
        self.extended_inputs = Some(inputs);
        self.extended_input_error = None;
        self.extended_inputs_loaded = true;
        Ok(())
    }
    pub fn load_extended_research_inputs(&mut self, path: &std::path::Path) -> Result<()> {
        self.set_extended_research_inputs(
            crate::ccm::extended_research::ExternalResearchInputs::from_file(path)?,
        )
    }
    /// Allow the requested diagnostics listed in `convergence_capture::lookahead`
    /// to be computed ahead of their turn (the default; see
    /// [`Self::set_lookahead_requests`]). Results, artifacts and their order
    /// are the same either way; this changes only when the numerical work runs.
    #[doc(hidden)]
    pub fn set_capture_lookahead(&mut self, enabled: bool) {
        self.lookahead_enabled = enabled;
        if !enabled {
            self.lookahead = None;
        }
    }
    /// Declare the diagnostics this run will request. Only declared
    /// diagnostics are computed ahead, so no work runs for a diagnostic the
    /// caller never requests; without a declaration nothing runs ahead.
    pub fn set_lookahead_requests(&mut self, ids: impl IntoIterator<Item = String>) {
        self.lookahead_requests = Some(ids.into_iter().collect());
    }
    /// Outcome-aware adapter. Missing external data remains Missing in a receipt;
    /// numerical nonacceptance stays in the retained measurement.
    pub fn capture_diagnostic_outcome(
        &mut self,
        id: &str,
        options: &CcmResearchCaptureOptions,
        cache: &ArtifactCacheContext<'_>,
    ) -> std::result::Result<CapturedDiagnostic, xc_cache::CaptureFailure> {
        if id == "target_comparison" && !crate::target::runtime_target_configured() {
            return Err(xc_cache::CaptureFailure::Missing {
                reason: "no runtime target is configured".into(),
            });
        }
        let result = self
            .capture_diagnostic(id, options, cache)
            .map_err(xc_cache::CaptureFailure::failed)?;
        if result.value["kind"] == "ccm_finite_diagnostic_analysis"
            && result.value_reference.is_none()
            && result.value["data"]["outcome"] == "blocked"
        {
            return Err(xc_cache::CaptureFailure::Blocked {
                reason: result.value["data"]["reason"]
                    .as_str()
                    .unwrap_or("diagnostic resource limit")
                    .into(),
            });
        }
        if result.value["kind"] != "ccm_finite_diagnostic_analysis"
            && result.value["data"]["outcome"] == "missing_input"
        {
            return Err(xc_cache::CaptureFailure::Missing {
                reason: result.value["data"]["reason"]
                    .as_str()
                    .unwrap_or("required retained inputs unavailable")
                    .into(),
            });
        }
        Ok(result)
    }

    fn prepare_run_once_inputs(
        &mut self,
        options: &CcmResearchCaptureOptions,
        cache: &ArtifactCacheContext<'_>,
    ) -> Result<()> {
        use crate::ccm::{convergence_capture::*, extended_research::*};
        if self.run_once_prepared {
            return Ok(());
        }
        let record = self
            .eigenpair
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("retained primary state unavailable"))?;
        let state = crate::ccm::state_geometry::RetainedState::from_payload(
            &record.manifest,
            &record.payload,
            std::slice::from_ref(&record.manifest.content_digest),
        )?;
        if let Some(error) = self
            .extended_inputs
            .as_ref()
            .and_then(|i| i.matches(&state).err())
        {
            self.extended_input_error = Some(error.to_string());
            self.extended_inputs = None;
            self.run_once_sources.clear();
        }
        let mut input=self.extended_inputs.clone().unwrap_or_else(|| serde_json::from_value(serde_json::json!({
            "schema_version":1,"source_eigenpair":record.manifest.content_digest,"lambda_squared":state.cutoff,"n_modes":state.modes,"precision_bits":state.precision,
            "convention_id":"toolkit_retained_run_inputs_v1","definition_digest":ContentDigest::sha256(b"toolkit_retained_run_inputs_v1"),"approximation_scope":"finite retained sources; automatic component actions and response reuse; no target formula"
        })).expect("static automatic input shape"));
        if std::env::var("XC_RESEARCH_PREPARE_TARGET_REFERENCE").is_ok_and(|v| v == "1") {
            let prepared = (|| -> Result<_> {
                let manifest = self.source.tau_manifest.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("retained Tau unavailable for research preparation")
                })?;
                let matrix = crate::ccm::retained_evidence::RetainedMatrix::from_admitted_runtime(
                    manifest.clone(),
                    state.cutoff.clone(),
                    state.modes,
                    state.precision,
                    &self.source.tau,
                )?;
                let prepared =
                    crate::ccm::research_prepare::prepare_arithmetic_inputs(&state, &matrix)?;
                Ok((prepared, manifest.clone()))
            })();
            match prepared {
                Ok((mut prepared, manifest)) => {
                    crate::ccm::research_completion::align_input_precisions(
                        &mut input,
                        &mut prepared,
                    )?;
                    if input.atoms.is_empty() {
                        input.atoms = prepared.atoms;
                        input.atom_coordinate = prepared.atom_coordinate;
                        input.atom_coverage = prepared.atom_coverage;
                    }
                    if input.energy_allowance.is_none() {
                        input.energy_allowance = prepared.energy_allowance;
                    }
                    let run_once = input.run_once.get_or_insert_with(Default::default);
                    if run_once.tail_form.is_none() {
                        run_once.tail_form =
                            prepared.run_once.as_mut().and_then(|r| r.tail_form.take());
                    }
                    self.run_once_sources.push(manifest);
                }
                Err(e) => input
                    .run_once
                    .get_or_insert_with(Default::default)
                    .producer_notes
                    .push(format!("automatic arithmetic preparation unavailable: {e}")),
            }
        }
        let mut derived = input.run_once.take().unwrap_or_default();
        let actual_orders = self.cfg.resolved_archimedean_orders(
            self.params.n_modes,
            &log_lambda_sq_hp(&self.params, self.cfg.precision_bits)?,
        )?;
        derived.producer_notes.push(format!("operator quadrature: policy={:?}; actual per-mode orders retained in ccm_archimedean_integrals; order_table_digest={}; minimum={}; maximum={}",
            self.cfg.research_quadrature_policy,ContentDigest::sha256(&serde_json::to_vec(&actual_orders)?).0,
            actual_orders.iter().min().unwrap(),actual_orders.iter().max().unwrap()));
        if self.extended_input_error.is_some() {
            derived.producer_notes.push("supplied external input rejected; source-only automatic diagnostics preserved; reference-dependent diagnostics remain failed".into());
        }
        if derived.component_actions.is_empty() && input.components.is_empty() {
            match self.automatic_component_actions(&state, cache) {
                Ok((actions, manifest)) => {
                    derived.component_actions = actions;
                    input.components_are_complete = true;
                    self.run_once_sources.push(manifest);
                }
                Err(error) => derived
                    .producer_notes
                    .push(format!("arithmetic components unavailable: {error}")),
            }
        }
        let mut completion = derived.completion.take().unwrap_or_default();
        if completion.independent_actions.is_empty() {
            let direct = (|| -> Result<_> {
                let tau = self
                    .source
                    .tau_manifest
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("Tau ancestry absent"))?;
                let dep = xc_cache::resolve_manifest_sources(tau, &self.run_once_sources, cache)?
                    .into_iter()
                    .find(|m| m.key.kind == "ccm_prime_component")
                    .ok_or_else(|| anyhow::anyhow!("direct prime parent absent"))?;
                let resolver = cache
                    .resolver
                    .ok_or_else(|| anyhow::anyhow!("exact source resolver absent"))?;
                let acceptance = cache
                    .acceptance
                    .ok_or_else(|| anyhow::anyhow!("source acceptance policy absent"))?;
                let _stage = crate::ccm::capture_runtime::Stage::new(
                    "direct prime read/validation/contraction",
                );
                let source = resolver.resolve_exact(
                    &dep.key,
                    &dep.content_digest,
                    CacheQuality::Validated,
                    acceptance,
                )?;
                let portable: PortablePrimeComponent = serde_json::from_slice(&source.payload)?;
                let matrix = decode_prime_component(&portable, &self.params, state.precision)?;
                let v = crate::ccm::extended_research::source_unit(&state, state.precision)?;
                let n = v.len();
                let action = matrix
                    .par_chunks(n)
                    .map(|row| -> Result<String> {
                        Ok(lossless_hp_decimal(
                            &(-crate::ccm::retained_evidence::dot(row, &v, state.precision)?),
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok((OperatorAction{label:"tau_prime_direct".into(),source_digest:source.manifest.content_digest.clone(),action,convention:"negative direct unsigned prime matrix action on the same signed unit state; exact retained Tau parent".into()},source.manifest))
            })();
            match direct {
                Ok((action, manifest)) => {
                    completion.independent_actions.push(action);
                    self.run_once_sources.push(manifest);
                }
                Err(e) => completion
                    .preparation_notes
                    .push(format!("independent prime check unavailable: {e}")),
            }
        }
        if completion.comparisons.is_empty() {
            match crate::ccm::research_cohort::discover(&state, cache) {
                Ok((snapshots, parents, notes)) => {
                    completion.comparisons = snapshots;
                    self.run_once_sources.extend(parents);
                    completion
                        .preparation_notes
                        .extend(notes.into_iter().take(64));
                }
                Err(e) => completion
                    .preparation_notes
                    .push(format!("cohort discovery unavailable: {e}")),
            }
        }
        if derived.derivative_actions.is_empty() {
            match self.capture_inner("u_flow_response", options, cache) {
                Ok(record) => {
                    let result = record
                        .value
                        .as_array()
                        .and_then(|v| v.first())
                        .ok_or_else(|| anyhow::anyhow!("u-flow artifact payload absent"));
                    match result.and_then(|v| {
                        Ok(serde_json::from_value::<PortableUFlowResponseAnalysis>(
                            v.clone(),
                        )?)
                    }) {
                        Ok(response) => {
                            let manifest = record
                                .sources
                                .iter()
                                .find(|m| m.key.kind == "ccm_u_flow_response_analysis")
                                .ok_or_else(|| anyhow::anyhow!("u-flow source identity absent"))?;
                            if response.eigenpair_content_digest != state.manifest.content_digest.0
                            {
                                bail!("u-flow source belongs to a different state");
                            }
                            let sign = crate::ccm::retained_evidence::orientation(
                                &state.coefficients,
                                state.precision,
                            );
                            for (j, root) in response.roots.iter().enumerate() {
                                if let Some(t) = &root.value {
                                    let total =
                                        response.channels.iter().find(|c| c.channel == "tau_total");
                                    completion.response_checks.push(
                                        crate::ccm::research_completion::ResponseCheck {
                                            ordinal: root
                                                .positive_root_index
                                                .unwrap_or(root.window_position + 1),
                                            t: t.clone(),
                                            source_digest: manifest.content_digest.clone(),
                                            branch: "production_shifted_secular".into(),
                                            coordinate: "mellin_t".into(),
                                            derivative_parameter: response
                                                .velocity_parameter
                                                .clone(),
                                            activation_convention: response
                                                .derivative_convention
                                                .clone(),
                                            fixed_velocity: total
                                                .and_then(|c| {
                                                    c.fixed_pole_root_velocity_responses.get(j)
                                                })
                                                .cloned()
                                                .flatten(),
                                            support_velocity: response
                                                .secular_pole_motion_root_velocity_responses
                                                .get(j)
                                                .cloned()
                                                .flatten(),
                                            total_velocity: response
                                                .total_moving_pole_root_velocity_responses
                                                .get(j)
                                                .cloned()
                                                .flatten(),
                                        },
                                    );
                                }
                            }
                            for channel in response.channels {
                                let action = channel
                                    .tau_velocity_action_on_state
                                    .iter()
                                    .map(|v| {
                                        Ok(lossless_hp_decimal(
                                            &(Float::with_val(state.precision, Float::parse(v)?)
                                                * sign),
                                        ))
                                    })
                                    .collect::<Result<Vec<_>>>()?;
                                derived.derivative_actions.push(OperatorAction {
                                    label: channel.channel,
                                    source_digest: manifest.content_digest.clone(),
                                    action,
                                    convention: response.derivative_convention.clone(),
                                });
                            }
                            derived.log_cutoff_velocity = Some("1".into());
                            self.run_once_sources.extend(record.sources);
                        }
                        Err(error) => derived
                            .producer_notes
                            .push(format!("u-flow source decode unavailable: {error}")),
                    }
                }
                Err(error) => derived
                    .producer_notes
                    .push(format!("u-flow response unavailable: {error}")),
            }
        }
        if input.cluster.is_empty() {
            match self.sectors(options, cache) {
                Ok(()) => {
                    let sectors = self.sectors.as_ref().expect("prepared sectors");
                    for (spec, manifest) in [
                        (&sectors.gap.even, &sectors.even_manifest),
                        (&sectors.gap.odd, &sectors.odd_manifest),
                    ] {
                        if let Some(manifest) = manifest {
                            self.run_once_sources.push(manifest.clone());
                            for vector in &spec.eigenpairs {
                                let full = match spec.parity {
                                    CcmParity::Even => expand_even_sector_vector(
                                        &vector.eigenvector,
                                        state.modes,
                                        state.precision,
                                    ),
                                    CcmParity::Odd => expand_odd_sector_vector(
                                        &vector.eigenvector,
                                        state.modes,
                                        state.precision,
                                    ),
                                };
                                input.cluster.push(ClusterVector {
                                    source_digest: manifest.content_digest.clone(),
                                    n_modes: state.modes,
                                    precision_bits: state.precision,
                                    eigenvalue: lossless_hp_decimal(&vector.eigenvalue),
                                    coefficients: full.iter().map(lossless_hp_decimal).collect(),
                                    assembly_policy: format!(
                                        "same retained Tau; {} sector; algebraic index {}",
                                        spec.parity.as_str(),
                                        vector.algebraic_index
                                    ),
                                });
                            }
                        }
                    }
                }
                Err(error) => derived
                    .producer_notes
                    .push(format!("sector inputs unavailable: {error}")),
            }
        }
        derived.completion = Some(completion);
        input.run_once = Some(derived);
        input.validate()?;
        self.extended_inputs = Some(input);
        self.run_once_prepared = true;
        Ok(())
    }

    fn automatic_component_actions(
        &self,
        state: &crate::ccm::state_geometry::RetainedState,
        cache: &ArtifactCacheContext<'_>,
    ) -> Result<(
        Vec<crate::ccm::convergence_capture::OperatorAction>,
        ArtifactManifest,
    )> {
        use crate::ccm::convergence_capture::OperatorAction;
        let p = state.precision;
        let l = log_lambda_sq_hp(&self.params, p)?;
        let tau = self
            .source
            .tau_manifest
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Tau provenance unavailable"))?;
        let parent = xc_cache::resolve_manifest_sources(tau, &self.run_once_sources, cache)?
            .into_iter()
            .find(|m| m.key.kind == "ccm_archimedean_integrals")
            .ok_or_else(|| anyhow::anyhow!("retained archimedean parent absent"))?;
        let resolver = cache
            .resolver
            .ok_or_else(|| anyhow::anyhow!("exact primitive resolver unavailable"))?;
        let policy = cache
            .acceptance
            .ok_or_else(|| anyhow::anyhow!("primitive acceptance policy unavailable"))?;
        let resolved = resolver.resolve_exact(
            &parent.key,
            &parent.content_digest,
            CacheQuality::Validated,
            policy,
        )?;
        let portable: PortableArchimedeanIntegrals = serde_json::from_slice(&resolved.payload)?;
        let integrals = decode_archimedean_integrals(&portable, &self.params, p)?;
        let manifest = resolved.manifest;
        if !xc_cache::manifest_depends_on(tau, &manifest)? {
            bail!("archimedean primitives do not match retained Tau ancestry");
        }
        let v = crate::ccm::extended_research::source_unit(state, p)?;
        let rows = matrix_point_math::component_actions(
            state.modes,
            &l,
            p,
            &integrals,
            &self.source.tau,
            &v,
        )?;
        let notes = "correctly rounded stored-matrix action on center-oriented unit coefficient state; pole and archimedean from exact retained primitive ancestry; prime=Tau-pole-archimedean is an algebraic reconstruction, not an independent prime-closure validation";
        let actions = ["tau_pole", "tau_archimedean", "tau_prime_reconstructed"]
            .iter()
            .enumerate()
            .map(|(j, label)| OperatorAction {
                label: (*label).into(),
                source_digest: if j == 1 {
                    manifest.content_digest.clone()
                } else {
                    tau.content_digest.clone()
                },
                action: rows
                    .iter()
                    .map(|x| {
                        lossless_hp_decimal(match j {
                            0 => &x.0,
                            1 => &x.1,
                            _ => &x.2,
                        })
                    })
                    .collect(),
                convention: notes.into(),
            })
            .collect();
        Ok((actions, manifest))
    }

    pub fn primary(&self) -> &HighPrecResult {
        &self.primary
    }
    /// Exact source identities for a durable application journal. Large
    /// numerical payloads remain in the configured artifact cache.
    pub fn primary_sources(&self) -> Vec<ArtifactManifest> {
        [
            &self.source.tau_manifest,
            &self.source.eigenpair_manifest,
            &self.source.secular_manifest,
            &self.source.root_manifest,
        ]
        .into_iter()
        .flatten()
        .cloned()
        .collect()
    }
    pub fn into_primary(self) -> HighPrecResult {
        self.primary
    }

    /// Decode the authenticated ordered even block of this run's Tau. A
    /// checkpoint eigenstate is supplied only for the actual even-sector route;
    /// natural/adaptive states are never replaced or projected for an export.
    pub fn retained_even_sources(
        &self,
        cache: &ArtifactCacheContext<'_>,
    ) -> Result<(
        super::super::prefix::RetainedEvenMatrix,
        Vec<super::super::prefix::RetainedEvenEigenpair>,
    )> {
        use super::super::prefix::{RetainedEvenEigenpair, RetainedEvenMatrix};
        let sink = RecordingSink::new(cache, &["ccm_even_sector_matrix"]);
        let observed = observing(cache, &sink);
        let mut tau = self.source.tau.clone();
        force_symmetric(&mut tau, self.params.matrix_size())?;
        let manifest = self
            .source
            .tau_manifest
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("retained Tau manifest missing"))?;
        xc_numerics::hp_runtime::run_hp(|| {
            resolve_even_sector_matrix_via_cache(&self.params, &self.cfg, &tau, manifest, &observed)
        })?;
        let record = sink
            .finish()?
            .pop()
            .ok_or_else(|| anyhow::anyhow!("retained even matrix missing"))?;
        let matrix = RetainedEvenMatrix::from_payload(
            &record.manifest,
            &record.payload,
            std::slice::from_ref(&record.manifest.content_digest),
        )?;
        let eigenpairs = if self.cfg.effective_parity_policy() == CcmParityPolicy::EvenSector {
            let record = self
                .eigenpair
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("retained eigenpair missing"))?;
            vec![RetainedEvenEigenpair::from_payload(
                &record.manifest,
                &record.payload,
                std::slice::from_ref(&record.manifest.content_digest),
            )?]
        } else {
            vec![]
        };
        Ok((matrix, eigenpairs))
    }

    fn sectors(
        &mut self,
        options: &CcmResearchCaptureOptions,
        cache: &ArtifactCacheContext<'_>,
    ) -> Result<()> {
        let options = options
            .sector_analysis
            .ok_or_else(|| anyhow::anyhow!("sector analysis not requested"))?;
        if self.sector_options.is_some_and(|prior| prior != options) {
            bail!("sector analysis options changed after capture; create a new retained run");
        }
        self.sector_options = Some(options);
        if let Some(error) = &self.sector_error {
            bail!("sector diagnostic unavailable: {error}");
        }
        if self.sectors.is_some() {
            return Ok(());
        }
        let sink = RecordingSink::new(cache, &["ccm_sector_gap", "ccm_sector_spectrum"]);
        let observed = observing(cache, &sink);
        let source = RetainedCcmSource {
            tau: self.source.tau.clone(),
            tau_manifest: self.source.tau_manifest.clone(),
            eigenpair_manifest: None,
            secular_manifest: None,
            root_manifest: None,
        };
        match analyze_sector_gap_from_retained_source(
            &self.params,
            &self.cfg,
            options,
            Some(&observed),
            source,
        ) {
            Ok(sectors) => {
                self.sector_record = Some(recorded(sink.finish()?)?);
                self.sectors = Some(sectors);
                Ok(())
            }
            Err(error) => {
                self.sector_error = Some(error.to_string());
                Err(error)
            }
        }
    }

    /// Resolve the run's sector-gap certificate once. The certificate capture
    /// and the research diagnostics that replay it share this single result.
    /// Like sector analysis, its sector and certification options are fixed by
    /// the first request; a different request requires a new retained run.
    fn ensure_sector_certificate(
        &mut self,
        options: &CcmResearchCaptureOptions,
        cache: &ArtifactCacheContext<'_>,
    ) -> Result<()> {
        let certification = options
            .sector_gap_certification
            .ok_or_else(|| anyhow::anyhow!("sector certification not requested"))?;
        if self
            .sector_certificate_options
            .is_some_and(|prior| prior != certification)
        {
            bail!(
                "sector-gap certification options changed after capture; create a new retained run"
            );
        }
        // Also rejects a changed sector request before any memoized result is used.
        self.sectors(options, cache)?;
        self.sector_certificate_options = Some(certification);
        if self.sector_certificate.is_some() {
            return Ok(());
        }
        if let Some(error) = &self.sector_certificate_error {
            bail!("sector-gap certificate unavailable: {error}");
        }
        let sectors = self.sectors.as_ref().expect("resolved sectors");
        let resolved = certify_sector_gap_from_resolution(
            &self.params,
            &self.cfg,
            certification,
            sectors,
            Some(cache),
        );
        match resolved {
            Ok(resolved) => {
                self.sector_certificate = Some(resolved);
                Ok(())
            }
            Err(error) => {
                self.sector_certificate_error = Some(format!("{error:#}"));
                Err(error)
            }
        }
    }

    /// Attempt one primary diagnostic. Callers convert errors to explicit
    /// receipt outcomes and continue other independent requests. Measurements
    /// include exact authenticated artifact manifests, on fresh and warm runs.
    /// Once sector analysis is attempted, its options are fixed for this retained
    /// run. A different count or route requires a new run, preventing reuse of
    /// sector-derived diagnostics under a different request.
    pub fn capture_diagnostic(
        &mut self,
        id: &str,
        options: &CcmResearchCaptureOptions,
        cache: &ArtifactCacheContext<'_>,
    ) -> Result<CapturedDiagnostic> {
        xc_numerics::hp_runtime::run_hp(|| self.capture_inner(id, options, cache))
    }
    /// Prepare configured source-bound inputs without capturing or publishing a diagnostic.
    /// The caller may extend the returned data, then set it back through the checked setter.
    #[doc(hidden)]
    pub fn prepare_extended_research_inputs(
        &mut self,
    ) -> Result<Option<crate::ccm::extended_research::ExternalResearchInputs>> {
        self.ensure_extended_input_sources()?;
        if let Some(error) = &self.extended_input_error {
            bail!("external research preparation: {error}");
        }
        Ok(self.extended_inputs.clone())
    }

    #[doc(hidden)]
    pub fn extended_research_sources(&self) -> &[ArtifactManifest] {
        &self.run_once_sources
    }

    /// Prepare signed finite components from this matrix's retained primitives.
    /// No primitive computation or eigenstate solve is permitted on a miss.
    #[doc(hidden)]
    pub fn retained_trial_components(
        &mut self,
        bound_precision_bits: u32,
        maximum_bytes: u64,
        cache: &ArtifactCacheContext<'_>,
    ) -> Result<
        Option<
            crate::ccm::convergence_capture::finite_capture::energy_extensions::ComponentRequest,
        >,
    > {
        use crate::ccm::convergence_capture::finite_capture::energy_extensions::{
            ComponentData, ComponentRequest, SignedComponent,
        };
        use rug::float::Round;
        use xc_solver::trial_energy::ExactBounds;
        anyhow::ensure!(
            (64..=4096).contains(&bound_precision_bits),
            "component enclosure precision outside budget"
        );
        let d = self.params.matrix_size();
        let entries = (d as u64).saturating_mul(d as u64 + 1) / 2;
        // Absolute dyadic enclosure grid 2^-bits; two endpoints per entry.
        // Positive exponent allowance is checked
        // before conversion below. Admission precedes dense primitive decoding.
        let estimate = entries.saturating_mul(3).saturating_mul(
            (4 * u64::from(bound_precision_bits) + 128)
                .saturating_mul(31)
                .div_ceil(100)
                + 100,
        );
        let policy = crate::ccm::capture_runtime::CaptureResourcePolicy::from_environment()?;
        let working = estimate.saturating_add(
            (d as u64)
                .saturating_mul(d as u64)
                .saturating_mul(4)
                .saturating_mul(u64::from(self.cfg.precision_bits).div_ceil(8) + 64),
        );
        if estimate > maximum_bytes.min(crate::ccm::capture_runtime::RESEARCH_INPUT_MAXIMUM_BYTES)
            || working > policy.maximum_working_bytes
        {
            return Ok(None);
        }
        let tau = self
            .source
            .tau_manifest
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("retained matrix identity absent"))?
            .clone();
        let parents = xc_cache::resolve_manifest_sources(&tau, &self.run_once_sources, cache)?;
        // Estimated admission, including raw retained payloads, temporary
        // decoded strings, and guarded assembly buffers; not a measured peak.
        let payload_bytes = parents
            .iter()
            .filter(|m| {
                matches!(
                    m.key.kind.as_str(),
                    "ccm_archimedean_integrals" | "ccm_prime_component"
                )
            })
            .fold(0u64, |total, m| total.saturating_add(m.size_bytes));
        if working
            .saturating_mul(2)
            .saturating_add(payload_bytes.saturating_mul(4))
            > policy.maximum_working_bytes
        {
            return Ok(None);
        }
        let resolver = cache
            .resolver
            .ok_or_else(|| anyhow::anyhow!("retained component resolver absent"))?;
        let acceptance = cache
            .acceptance
            .ok_or_else(|| anyhow::anyhow!("retained component acceptance policy absent"))?;
        let mut resolved = Vec::new();
        for kind in ["ccm_archimedean_integrals", "ccm_prime_component"] {
            let parent = parents
                .iter()
                .find(|m| m.key.kind == kind)
                .ok_or_else(|| anyhow::anyhow!("retained component parent absent: {kind}"))?;
            let record = resolver.resolve_exact(
                &parent.key,
                &parent.content_digest,
                CacheQuality::Validated,
                acceptance,
            )?;
            anyhow::ensure!(
                xc_cache::manifest_depends_on(&tau, &record.manifest)?,
                "component ancestry mismatch"
            );
            resolved.push(record);
        }
        let p = self.cfg.precision_bits;
        let integrals = decode_archimedean_integrals(
            &serde_json::from_slice(&resolved[0].payload)?,
            &self.params,
            p,
        )?;
        let prime = decode_prime_component(
            &serde_json::from_slice(&resolved[1].payload)?,
            &self.params,
            p,
        )?;
        let l = log_lambda_sq_hp(&self.params, p)?;
        let (pole, arch) =
            assemble_pole_and_archimedean_components(self.params.n_modes, &l, p, &integrals)?;
        let mut components = Vec::new();
        for (label, sign, data) in [
            ("stored_pole", "1", pole),
            ("stored_archimedean", "-1", arch),
            ("stored_prime", "-1", prime),
        ] {
            anyhow::ensure!(
                data.iter()
                    .all(|x| x.is_finite() && x.get_exp().is_none_or(|e| e <= 64)),
                "component exponent exceeds serialized admission estimate"
            );
            let mut values = Vec::with_capacity(d * (d + 1) / 2);
            for i in 0..d {
                for j in i..d {
                    let scaled = Float::with_val(p, &data[i * d + j]) << bound_precision_bits;
                    let denominator = rug::Integer::from(1) << bound_precision_bits;
                    let lo = scaled
                        .to_integer_round(Round::Down)
                        .ok_or_else(|| anyhow::anyhow!("component underflow"))?
                        .0;
                    let hi = scaled
                        .to_integer_round(Round::Up)
                        .ok_or_else(|| anyhow::anyhow!("component overflow"))?
                        .0;
                    values.push(ExactBounds {
                        lower: rug::Rational::from((lo, denominator.clone())).to_string(),
                        upper: rug::Rational::from((hi, denominator)).to_string(),
                    });
                }
            }
            components.push(SignedComponent::from_data(ComponentData {
                label: label.into(),
                signed_weight: sign.into(),
                diagonal: vec![],
                upper_triangle: values,
                rank_one: vec![],
            })?);
        }
        let request = ComponentRequest {
            matrix_digest: tau.content_digest,
            basis_id: "centered_full_V_fourier".into(),
            components,
            assembly_operator_norm_error: None,
        };
        anyhow::ensure!(
            serde_json::to_vec(&request)?.len() as u64
                <= maximum_bytes.min(crate::ccm::capture_runtime::RESEARCH_INPUT_MAXIMUM_BYTES),
            "component serialization exceeded admitted size"
        );
        for record in resolved {
            if !self
                .run_once_sources
                .iter()
                .any(|m| m.content_digest == record.manifest.content_digest)
            {
                self.run_once_sources.push(record.manifest);
            }
        }
        Ok(Some(request))
    }

    #[doc(hidden)]
    pub fn prepare_retained_trial_components(
        &mut self,
        bound_precision_bits: u32,
        maximum_bytes: u64,
        cache: &ArtifactCacheContext<'_>,
    ) -> Result<bool> {
        let mut input = self.prepare_extended_research_inputs()?.ok_or_else(|| {
            anyhow::anyhow!("component preparation requires retained research inputs")
        })?;
        if input
            .finite_diagnostics
            .as_ref()
            .is_some_and(|i| i.component_energy.is_some())
        {
            return Ok(true);
        }
        let Some(request) =
            self.retained_trial_components(bound_precision_bits, maximum_bytes, cache)?
        else {
            return Ok(false);
        };
        let parents = self.run_once_sources.clone();
        let extra = input.finite_diagnostics.get_or_insert_with(|| {
            crate::ccm::convergence_capture::finite_capture::Inputs {
                scope: "retained signed component enclosures; stored primitive and matrix scope"
                    .into(),
                ..Default::default()
            }
        });
        extra.component_energy = Some(request);
        self.set_extended_research_inputs(input)?;
        // Only this component field changed. Preserve its authenticated parents
        // across the general setter's invalidation of derived input state.
        self.run_once_sources = parents;
        Ok(true)
    }

    fn ensure_extended_input_sources(&mut self) -> Result<()> {
        use crate::ccm::extended_research::ExternalResearchInputs;
        use crate::ccm::state_geometry::RetainedState;
        if !self.extended_inputs_loaded {
            self.extended_inputs_loaded = true;
            if let Some(path) = std::env::var_os("XC_RESEARCH_INPUTS_FILE") {
                match ExternalResearchInputs::from_file(std::path::Path::new(&path)) {
                    Ok(input) => self.extended_inputs = Some(input),
                    Err(e) => self.extended_input_error = Some(e.to_string()),
                }
            }
            if self.extended_inputs.is_none()
                && self.extended_input_error.is_none()
                && std::env::var_os("XC_RESEARCH_REFERENCE_FILE").is_none()
                && std::env::var("XC_RESEARCH_PREPARE_TARGET_REFERENCE").is_ok_and(|v| v == "1")
            {
                let prepared = (|| -> Result<_> {
                    let record = self
                        .eigenpair
                        .as_ref()
                        .ok_or_else(|| anyhow::anyhow!("retained primary state unavailable"))?;
                    let state = RetainedState::from_payload(
                        &record.manifest,
                        &record.payload,
                        std::slice::from_ref(&record.manifest.content_digest),
                    )?;
                    let spec = crate::target::TargetProfileSpec::from_environment()?;
                    crate::ccm::research_target::prepare(
                        &state,
                        &spec,
                        (16 * state.modes).clamp(256, 131072),
                    )
                })();
                match prepared {
                    Ok((reference, input)) => {
                        self.prepared_reference = Some(
                            crate::ccm::research_completion::ReferencePreparation {
                                schema_version: 1,
                                finite_reference: Some(reference.clone()),
                                sampled_reference: input.target.clone(),
                                lambda_squared: input.lambda_squared.clone(),
                                precision_bits: input.precision_bits,
                                definition_digest: input.definition_digest.clone(),
                                approximation_scope: format!(
                                    "{}; signed jets refer to this explicitly finite Fourier projection, extended by zero outside the run window",
                                    input.approximation_scope
                                ),
                                weighted_atoms: vec![],
                                atom_coordinate: None,
                                atom_coverage: None,
                                tail_form: None,
                                tail_recipe: None,
                                completion: None,
                            },
                        );
                        if self.research_inputs.is_none() {
                            self.research_inputs = Some(reference);
                        }
                        self.extended_inputs = Some(input);
                    }
                    Err(e) => {
                        self.extended_input_error =
                            Some(format!("runtime target research preparation: {e}"))
                    }
                }
            }
        }
        if self.prepared_reference.is_none() {
            if let Some(path) = std::env::var_os("XC_RESEARCH_REFERENCE_FILE") {
                match crate::ccm::research_completion::ReferencePreparation::from_file(
                    std::path::Path::new(&path),
                ) {
                    Ok(reference) => {
                        if let Some(finite) = &reference.finite_reference {
                            self.research_inputs = Some(finite.clone());
                        }
                        self.prepared_reference = Some(reference);
                    }
                    Err(e) => {
                        self.extended_input_error = Some(format!("reference preparation: {e}"))
                    }
                }
            }
        }
        if let Some(reference) = &self.prepared_reference {
            if self.extended_inputs.is_none() {
                if let Some(record) = &self.eigenpair {
                    let state = crate::ccm::state_geometry::RetainedState::from_payload(
                        &record.manifest,
                        &record.payload,
                        std::slice::from_ref(&record.manifest.content_digest),
                    )?;
                    match reference.prepare(&state, None) {
                        Ok(i) => self.extended_inputs = Some(i),
                        Err(e) => {
                            self.extended_input_error = Some(format!("reference preparation: {e}"))
                        }
                    }
                }
            }
        }
        Ok(())
    }
    /// Start the look-ahead diagnostics not yet requested on one background
    /// thread, once per run. It starts after a capture was computed rather
    /// than reused (the run is cold) and once the run's inputs are settled:
    /// run-once preparation done and any reference jets added. This is
    /// scheduling only. Each result is used where its serial computation
    /// runs, and only if the inputs it was computed from equal the inputs
    /// there; otherwise that diagnostic is computed inline, as before. The
    /// look-ahead copy of Tau and a second concurrent workspace must fit
    /// within half the declared working budget.
    fn start_lookahead(&mut self, state: &crate::ccm::state_geometry::RetainedState) {
        use crate::ccm::convergence_capture::{finite_capture::ExecutionContext, lookahead};
        if !self.lookahead_enabled
            || self.lookahead_started
            || !self.run_once_prepared
            || (self.prepared_reference.is_some() && !self.prepared_reference_jets)
        {
            return;
        }
        let ids = lookahead_ids(self.lookahead_requests.as_ref(), &self.lookahead_reached);
        if ids.is_empty() {
            return;
        }
        self.lookahead_started = true;
        let jobs = (|| -> Result<_> {
            let policy = crate::ccm::capture_runtime::CaptureResourcePolicy::from_environment()?;
            let entry_bytes = self
                .source
                .tau
                .iter()
                .map(|x| 32 + u64::from(x.prec()).div_ceil(64) * 8)
                .sum::<u64>();
            // Tau entries (owned copy and workspace) and one copy of the
            // external input, which the jobs share when both roles use it.
            let input_bytes = self
                .extended_inputs
                .as_ref()
                .map(|input| serde_json::to_vec(input).map(|bytes| bytes.len() as u64))
                .transpose()?
                .unwrap_or(0);
            if entry_bytes
                .saturating_mul(6)
                .saturating_add(input_bytes.saturating_mul(2))
                > policy.maximum_working_bytes
            {
                bail!("look-ahead exceeds the declared working budget");
            }
            let manifest = self
                .source
                .tau_manifest
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("retained Tau manifest missing"))?;
            let matrix = crate::ccm::retained_evidence::RetainedMatrix::from_admitted_runtime(
                manifest.clone(),
                state.cutoff.clone(),
                state.modes,
                state.precision,
                &self.source.tau,
            )?;
            // Exactly the arguments the capture paths pass: finite
            // diagnostics omit a failed input; the listed retained
            // convergence diagnostics are complete, take no roots and use
            // the effective options of their capture.
            let finite_input = if self.extended_input_error.is_some() {
                None
            } else {
                self.extended_inputs.as_ref()
            };
            let execution = ExecutionContext {
                policy: policy.clone(),
                certificate_error: self.sector_certificate_error.clone(),
                input_preparation_error: self.extended_input_error.clone(),
            };
            let options = crate::ccm::extended_research::effective_options(
                &self.extension_options("finite_section_transfer", true, state, None)?,
                &policy,
            );
            lookahead::jobs(
                &ids,
                state,
                &matrix,
                finite_input,
                &execution,
                self.extended_inputs.as_ref(),
                &options,
            )
        })();
        if let Ok(jobs) = jobs {
            self.lookahead = crate::ccm::capture_runtime::LookAhead::start(jobs);
        }
    }

    /// Options of an extended capture of `id` (after the `_full` suffix is
    /// removed) before the capture applies its working budget.
    fn extension_options(
        &self,
        id: &str,
        complete: bool,
        state: &crate::ccm::state_geometry::RetainedState,
        roots: Option<&crate::ccm::retained_evidence::RetainedRoots>,
    ) -> Result<crate::ccm::extended_research::ExtensionOptions> {
        let mut options = self
            .extended_options
            .clone()
            .unwrap_or_else(|| crate::ccm::extended_research::ExtensionOptions::for_source(state));
        if self.extended_options.is_none() {
            options.working_precision_bits = options
                .working_precision_bits
                .max(
                    self.extended_inputs
                        .as_ref()
                        .map_or(0, |i| i.precision_bits.saturating_add(64)),
                )
                .max(roots.map_or(0, |r| r.dataset.precision_bits.saturating_add(64)));
        }
        if complete {
            if self.extended_options.is_none() {
                let policy =
                    crate::ccm::capture_runtime::CaptureResourcePolicy::from_environment()?;
                options.maximum_estimated_output_bytes = policy.maximum_output_bytes;
                options.maximum_working_bytes = Some(policy.maximum_working_bytes);
            }
            if matches!(id, "directional_response" | "root_transport") {
                options.maximum_directional_rows = roots.map_or(0, |r| r.dataset.points.len());
            }
        }
        Ok(options)
    }
    fn capture_inner(
        &mut self,
        id: &str,
        options: &CcmResearchCaptureOptions,
        cache: &ArtifactCacheContext<'_>,
    ) -> Result<CapturedDiagnostic> {
        if id == "u_flow_response" {
            if let Some(saved) = &self.uflow_capture {
                return Ok(saved.clone());
            }
        }
        let complete = id.ends_with("_full")
            || crate::ccm::convergence_capture::DIAGNOSTICS.contains(&id)
            || crate::ccm::research_completion::DIAGNOSTICS.contains(&id);
        let id = id.strip_suffix("_full").unwrap_or(id);
        if crate::ccm::capture::FINITE_DIAGNOSTICS.contains(&id)
            || crate::ccm::extended_research::DIAGNOSTICS.contains(&id)
            || crate::ccm::convergence_capture::DIAGNOSTICS.contains(&id)
            || crate::ccm::research_completion::DIAGNOSTICS.contains(&id)
        {
            use crate::ccm::extended_research::*;
            use crate::ccm::retained_evidence::{RetainedMatrix, RetainedRoots};
            use crate::ccm::state_geometry::RetainedState;
            // Requested here: never computed ahead from now on.
            self.lookahead_reached.insert(id.to_owned());
            let mut certificate_error = self.sector_certificate_error.clone();
            // The run's own certificate, when requested, supplies the source
            // assembly certificate these two diagnostics replay.
            if matches!(
                id,
                "capture_preflight" | "transform_enclosure" | "finite_root_budget"
            ) && options.sector_gap_certification.is_some()
            {
                if let Err(error) = self.ensure_sector_certificate(options, cache) {
                    certificate_error = Some(format!("{error:#}"));
                    xc_core::progress_message!(
                        "  sector-gap certificate unavailable to {id}: {error:#}"
                    );
                }
            }
            self.ensure_extended_input_sources()?;
            if complete {
                self.prepare_run_once_inputs(options, cache)?;
            }
            // Source-only groups remain available even when optional input loading failed.
            if !matches!(
                id,
                "compactness"
                    | "finite_root_budget"
                    | "trial_vector_energy"
                    | "trial_vector_parity"
                    | "indexed_prolate_comparison"
                    | "directional_error_bound"
                    | "finite_tail_bound"
                    | "spectral_cluster_bound"
                    | "dimension_precision_budget"
                    | "arithmetic_energy"
                    | "spectral_cluster"
                    | "directional_response"
                    | "resolution_budget"
                    | "complex_transform"
                    | "root_transport"
                    | "operator_cluster"
                    | "finite_section_transfer"
                    | "observable_budget"
                    | "capture_preflight"
                    | "consistency"
                    | "configuration_comparison"
                    | "transform_enclosure"
            ) {
                if let Some(error) = &self.extended_input_error {
                    bail!("external research inputs invalid: {error}");
                }
            }
            let record = self
                .eigenpair
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("retained eigenpair unavailable"))?;
            let state = RetainedState::from_payload(
                &record.manifest,
                &record.payload,
                std::slice::from_ref(&record.manifest.content_digest),
            )?;
            let matrix = if matches!(
                id,
                "trial_vector_energy"
                    | "trial_vector_parity"
                    | "directional_error_bound"
                    | "finite_tail_bound"
                    | "spectral_cluster_bound"
                    | "arithmetic_energy"
                    | "directional_response"
                    | "root_transport"
                    | "operator_cluster"
                    | "finite_section_transfer"
                    | "capture_preflight"
                    | "configuration_comparison"
            ) {
                self.source
                    .tau_manifest
                    .as_ref()
                    .map(|manifest| {
                        RetainedMatrix::from_admitted_runtime(
                            manifest.clone(),
                            state.cutoff.clone(),
                            state.modes,
                            state.precision,
                            &self.source.tau,
                        )
                    })
                    .transpose()?
            } else {
                None
            };
            let roots = if matches!(
                id,
                "finite_root_budget"
                    | "directional_error_bound"
                    | "dimension_precision_budget"
                    | "directional_response"
                    | "resolution_budget"
                    | "complex_transform"
                    | "root_transport"
                    | "observable_budget"
                    | "capture_preflight"
                    | "configuration_comparison"
                    | "transform_enclosure"
                    | "signed_transform"
                    | "weighted_reference_projection"
            ) {
                match (&self.source.root_manifest, &self.source.secular_manifest) {
                    (Some(r), Some(m)) => {
                        let root = self.retained_records.iter().find(|a| {
                            a.manifest.key == r.key && a.manifest.content_digest == r.content_digest
                        });
                        let secular = self.retained_records.iter().find(|a| {
                            a.manifest.key == m.key && a.manifest.content_digest == m.content_digest
                        });
                        match (root, secular) {
                            (Some(root), Some(secular)) => Some(RetainedRoots::from_payload(
                                r,
                                &root.payload,
                                m,
                                &secular.payload,
                                &state,
                                &[r.content_digest.clone(), m.content_digest.clone()],
                            )?),
                            _ => None,
                        }
                    }
                    _ => None,
                }
            } else {
                None
            };
            if !self.prepared_reference_jets && roots.is_some() {
                if let Some(reference) = &self.prepared_reference {
                    match reference.prepare(&state, roots.as_ref()) {
                        Ok(mut prepared) => {
                            if let Some(existing) = &mut self.extended_inputs {
                                crate::ccm::research_completion::align_input_precisions(
                                    existing,
                                    &mut prepared,
                                )?;
                                if existing.reference_jets.is_empty() {
                                    existing.reference_jets = prepared.reference_jets;
                                }
                            } else {
                                self.extended_inputs = Some(prepared);
                            }
                            self.prepared_reference_jets = true;
                        }
                        Err(e) => {
                            self.extended_input_error =
                                Some(format!("reference jet preparation: {e}"))
                        }
                    }
                }
            }
            let options = self.extension_options(id, complete, &state, roots.as_ref())?;
            let input = if id == "compactness"
                || (crate::ccm::capture::FINITE_DIAGNOSTICS.contains(&id)
                    && self.extended_input_error.is_some())
            {
                None
            } else {
                self.extended_inputs.as_ref()
            };
            let certificate = self
                .sector_certificate
                .as_ref()
                .filter(|_| certificate_error.is_none())
                .filter(|_| {
                    matches!(
                        id,
                        "capture_preflight" | "transform_enclosure" | "finite_root_budget"
                    )
                })
                .and_then(|resolved| {
                    resolved
                        .produced_manifest
                        .as_ref()
                        .or(resolved.reused_manifest.as_ref())
                        .map(|manifest| (&resolved.value, manifest))
                });
            use crate::ccm::convergence_capture::lookahead;
            let (captured, produced) = if crate::ccm::capture::FINITE_DIAGNOSTICS.contains(&id) {
                use crate::ccm::convergence_capture::finite_capture::{
                    capture_with_prepared, ExecutionContext, Prepared,
                };
                let execution = ExecutionContext {
                    policy: crate::ccm::capture_runtime::CaptureResourcePolicy::from_environment()?,
                    certificate_error,
                    input_preparation_error: self.extended_input_error.clone(),
                };
                let (state_ref, matrix_ref, roots_ref) = (&state, matrix.as_ref(), roots.as_ref());
                let execution_ref = &execution;
                let prepared = self.lookahead.as_ref().map(|lane| -> Prepared<'_> {
                    Box::new(move || {
                        lookahead::claim_finite(
                            lane,
                            id,
                            state_ref,
                            matrix_ref,
                            roots_ref.is_some(),
                            input,
                            certificate.is_some(),
                            execution_ref,
                        )
                    })
                });
                let result = capture_with_prepared(
                    id,
                    &state,
                    matrix.as_ref(),
                    roots.as_ref(),
                    input,
                    certificate,
                    &self.run_once_sources,
                    cache,
                    &execution,
                    prepared,
                )?;
                let produced = result.produced_manifest.is_some();
                (CapturedDiagnostic::from_cached(result)?, produced)
            } else {
                let (state_ref, matrix_ref) = (&state, matrix.as_ref());
                let prepared = self.lookahead.as_ref().map(|lane| -> ExtendedPrepared<'_> {
                    Box::new(move |options, plain| {
                        lookahead::claim_extended(
                            lane, id, state_ref, matrix_ref, input, options, plain,
                        )
                    })
                });
                let result = capture_extended_prepared(
                    id,
                    &state,
                    matrix.as_ref(),
                    roots.as_ref(),
                    input,
                    certificate,
                    &options,
                    &self.run_once_sources,
                    cache,
                    prepared,
                )?;
                let produced = result.produced_manifest.is_some();
                (CapturedDiagnostic::from_cached(result)?, produced)
            };
            if produced {
                self.start_lookahead(&state);
            }
            return Ok(captured);
        }
        if matches!(
            id,
            "indexed_transform" | "operator_energy" | "root_band" | "reference_projection"
        ) {
            use crate::ccm::retained_evidence::*;
            use crate::ccm::state_geometry::RetainedState;
            let record = self
                .eigenpair
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("retained primary eigenpair missing"))?;
            let state = RetainedState::from_payload(
                &record.manifest,
                &record.payload,
                std::slice::from_ref(&record.manifest.content_digest),
            )?;
            if id == "operator_energy" {
                let manifest = self
                    .source
                    .tau_manifest
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("retained Tau manifest missing"))?;
                let matrix = RetainedMatrix::from_admitted_runtime(
                    manifest.clone(),
                    state.cutoff.clone(),
                    state.modes,
                    state.precision,
                    &self.source.tau,
                )?;
                return Ok(CapturedDiagnostic::from_cached(capture_operator_energy(
                    &state, &matrix, cache,
                )?)?);
            }
            if id == "reference_projection" {
                if self.research_inputs.is_none() {
                    if let Some(path) = std::env::var_os("XC_RESEARCH_REFERENCE_FILE") {
                        let reference =
                            crate::ccm::research_completion::ReferencePreparation::from_file(
                                std::path::Path::new(&path),
                            )?;
                        self.research_inputs = reference.finite_reference;
                    }
                }
                let Some(inputs) = self.research_inputs.as_ref() else {
                    return Ok(CapturedDiagnostic::new(
                        &serde_json::json!({"kind":"ccm_reference_projection_analysis","data":{"outcome":"missing_input","reason":"finite Fourier reference unavailable; supply ResearchInputs or XC_RESEARCH_REFERENCE_FILE; sampled-only references still support weighted projection"}}),
                        vec![],
                    )?);
                };
                return capture_configured_projection(&state, inputs, cache);
            }
            let root_manifest = self
                .source
                .root_manifest
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("retained root window missing"))?;
            let secular_manifest = self
                .source
                .secular_manifest
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("retained secular source missing"))?;
            let root = self
                .retained_records
                .iter()
                .find(|r| {
                    r.manifest.key == root_manifest.key
                        && r.manifest.content_digest == root_manifest.content_digest
                })
                .ok_or_else(|| anyhow::anyhow!("retained root bytes missing"))?;
            let secular = self
                .retained_records
                .iter()
                .find(|r| {
                    r.manifest.key == secular_manifest.key
                        && r.manifest.content_digest == secular_manifest.content_digest
                })
                .ok_or_else(|| anyhow::anyhow!("retained secular bytes missing"))?;
            let roots = RetainedRoots::from_payload(
                root_manifest,
                &root.payload,
                secular_manifest,
                &secular.payload,
                &state,
                &[
                    root_manifest.content_digest.clone(),
                    secular_manifest.content_digest.clone(),
                ],
            )?;
            if id == "root_band" {
                return Ok(CapturedDiagnostic::from_cached(capture_root_window(
                    &roots, cache,
                )?)?);
            }
            return Ok(CapturedDiagnostic::from_cached(
                capture_transforms_at_roots(
                    &state,
                    &roots,
                    &TransformOptions::for_roots(&state, &roots),
                    cache,
                )?,
            )?);
        }
        if id == "checkpoint_spectra" {
            let (matrix, _) = self.retained_even_sources(cache)?;
            let ladder = crate::ccm::capture::checkpoint_spectrum_ladder(matrix.dimension());
            let result = super::checkpoint_low_spectra_via_cache(&matrix, &ladder, 3, cache)?;
            return Ok(CapturedDiagnostic::from_cached(result)?);
        }
        if id == "target_comparison" {
            // Hard private-only: derived from the private runtime target.
            let _stage = crate::ccm::capture_runtime::Stage::new(format!("{id} compute"));
            let (matrix, _) = self.retained_even_sources(cache)?;
            return Ok(CapturedDiagnostic::from_cached(
                crate::distance::hp::capture_target_comparison_via_cache(
                    &self.params,
                    &self.cfg,
                    &matrix,
                    cache,
                )?,
            )?);
        }
        if id == "assembly_error" {
            #[cfg(not(feature = "arb"))]
            bail!("assembly error analysis requires the xc-spectral arb feature");
            #[cfg(feature = "arb")]
            {
                // Sector enclosures add exact-form eigenvalue bounds; a sector
                // limitation only omits those bounds from this measurement.
                if options.sector_analysis.is_some() {
                    let _ = self.sectors(options, cache);
                }
                let tau_manifest = self.source.tau_manifest.clone().ok_or_else(|| {
                    anyhow::anyhow!("retained Tau manifest unavailable for assembly error analysis")
                })?;
                let eigenpair = self.source.eigenpair_manifest.clone();
                let sectors = self.sectors.as_ref().and_then(|resolution| {
                    resolution
                        .gap_manifest
                        .as_ref()
                        .map(|manifest| (&resolution.gap, manifest))
                });
                let result = super::assembly_error::resolve_assembly_error_via_cache(
                    &self.params,
                    &self.cfg,
                    &self.source.tau,
                    &tau_manifest,
                    eigenpair.as_ref().map(|manifest| (&self.primary, manifest)),
                    sectors,
                    cache,
                )?;
                return Ok(CapturedDiagnostic::from_cached(result)?);
            }
        }
        if id == "state_geometry" {
            use crate::ccm::state_geometry::{
                analyze_state_geometry_via_cache, GeometryOptions, RetainedState,
            };
            let record = self.eigenpair.as_ref().ok_or_else(|| {
                anyhow::anyhow!("retained primary eigenpair unavailable for state geometry")
            })?;
            let state = RetainedState::from_payload(
                &record.manifest,
                &record.payload,
                std::slice::from_ref(&record.manifest.content_digest),
            )?;
            let result = analyze_state_geometry_via_cache(
                &state,
                &GeometryOptions::for_source(&state),
                cache,
            )?;
            return Ok(CapturedDiagnostic::from_cached(result)?);
        }
        if id == "sector_gap_certificate" {
            // Recorded by reference to the certificate artifact, like the root
            // certificate, so the receipt does not carry a second copy.
            self.ensure_sector_certificate(options, cache)?;
            let resolved = self
                .sector_certificate
                .as_ref()
                .expect("resolved certificate");
            return Ok(
                match resolved
                    .produced_manifest
                    .as_ref()
                    .or(resolved.reused_manifest.as_ref())
                {
                    Some(manifest) => CapturedDiagnostic::by_reference(
                        &resolved.value,
                        vec![manifest.clone()],
                        false,
                        vec![manifest.clone()],
                    )?,
                    None => CapturedDiagnostic::new(
                        &resolved.value,
                        self.sector_record
                            .as_ref()
                            .map(|record| record.sources.clone())
                            .unwrap_or_default(),
                    )?,
                },
            );
        }
        if matches!(id, "evenness" | "sector_analysis") {
            self.sectors(options, cache)?;
            let sectors = self.sectors.as_ref().expect("resolved sectors");
            let record = self.sector_record.as_ref().expect("recorded sectors");
            if id == "evenness" {
                let value =
                    evenness_from_sector_gap(&self.params, self.cfg.precision_bits, &sectors.gap)?;
                return Ok(CapturedDiagnostic::new(
                    &serde_json::json!({"method":"resolved_stored_parity_sector_lift_v2", "claim_scope":value.claim_scope, "assembly_error_bound":null, "evenness_deviation":value.evenness_deviation.to_string(), "natural_eigenvalue":value.natural_eigenvalue.to_string(), "forced_eigenvalue":value.forced_eigenvalue.to_string()}),
                    record.sources.clone(),
                )?);
            }
            return Ok(CapturedDiagnostic::new(
                &record.value,
                record.sources.clone(),
            )?);
        }
        let kind = match id {
            "root_conditioning" => "ccm_root_conditioning_analysis",
            "prime_power_response" => "ccm_prime_power_response_analysis",
            "u_flow_response" => "ccm_u_flow_response_analysis",
            "distance_profile" => "ccm_eigenfunction_profile",
            "target_distance" => "ccm_target_distance",
            "distance_resolution" => "ccm_distance_resolution_evidence",
            "target_residual_analysis" => "ccm_target_residual_analysis",
            "deviation_decomposition" => "ccm_deviation_decomposition",
            "root_certificate" => "ccm_root_certification_report",
            _ => bail!("unknown retained-run diagnostic {id:?}"),
        };
        let sink = RecordingSink::new(cache, std::slice::from_ref(&kind));
        let observed = observing(cache, &sink);
        let p = &self.params;
        let cfg = &self.cfg;
        let primary = &self.primary;
        let source = &self.source;
        let required = |m: &Option<ArtifactManifest>| {
            m.clone()
                .ok_or_else(|| anyhow::anyhow!("required primary source manifest missing"))
        };
        let mut distance_resolution_verdicts = None;
        if matches!(id, "prime_power_response" | "u_flow_response") {
            if cfg.effective_parity_policy() != CcmParityPolicy::EvenSector {
                bail!("response requires an isolated even-sector state; primary parity preserved");
            }
            let l = log_lambda_sq_hp(p, cfg.precision_bits)?;
            let args = (
                required(&source.tau_manifest)?,
                required(&source.eigenpair_manifest)?,
                required(&source.root_manifest)?,
                required(&source.secular_manifest)?,
            );
            if id == "prime_power_response" {
                resolve_prime_power_response_analysis_via_cache(
                    p,
                    cfg,
                    &l,
                    &source.tau,
                    &primary.weil_min_eigenvalue,
                    &primary.xi,
                    &primary.eigenvalues_pos,
                    primary.first_positive_root_index,
                    &args.0,
                    &args.1,
                    &args.2,
                    &args.3,
                    &observed,
                )?;
            } else {
                resolve_u_flow_response_analysis_via_cache(
                    p,
                    cfg,
                    &l,
                    &source.tau,
                    &primary.weil_min_eigenvalue,
                    &primary.xi,
                    &primary.eigenvalues_pos,
                    primary.first_positive_root_index,
                    &args.0,
                    &args.1,
                    &args.2,
                    &args.3,
                    &observed,
                )?;
            }
        } else if id == "root_conditioning" {
            resolve_root_conditioning_analysis_via_cache(
                p,
                cfg,
                &log_lambda_sq_hp(p, cfg.precision_bits)?,
                &primary.xi,
                &primary.eigenvalues_pos,
                primary.first_positive_root_index,
                &required(&source.root_manifest)?,
                &required(&source.secular_manifest)?,
                &observed,
            )?;
        } else if id == "root_certificate" {
            let certification = options
                .root_certification
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("root certification not requested"))?;
            // Every requested root receives a row: certified enclosures where
            // certifiable, computed values with reasons everywhere else.
            let (report, manifest) = super::resolve_root_certification_report_via_cache(
                p,
                cfg,
                primary,
                &primary.xi,
                Some(&required(&source.root_manifest)?),
                Some(&required(&source.secular_manifest)?),
                certification,
                Some(&observed),
            )?;
            let mut sources = vec![
                required(&source.root_manifest)?,
                required(&source.secular_manifest)?,
            ];
            return Ok(match manifest {
                Some(manifest) => {
                    sources.push(manifest.clone());
                    CapturedDiagnostic::by_reference(&report, vec![manifest], false, sources)?
                }
                None => CapturedDiagnostic::new(&report, sources)?,
            });
        } else if id == "distance_profile" {
            let distance = options
                .distance_capture
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("profile capture not requested"))?;
            let rule = distance
                .rules
                .first()
                .ok_or_else(|| anyhow::anyhow!("profile capture requires a grid convention"))?;
            let _stage = crate::ccm::capture_runtime::Stage::new(format!("{id} compute"));
            crate::distance::hp::capture_ccm_profile_via_cache(
                p,
                cfg,
                distance.profile_steps,
                rule.variable(),
                &observed,
            )?;
        } else {
            let distance = options
                .distance_capture
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("distance capture not requested"))?;
            let alpha = Float::with_val(cfg.precision_bits, Float::parse(&distance.alpha)?);
            let _stage = crate::ccm::capture_runtime::Stage::new(format!("{id} compute"));
            let captured = crate::distance::hp::capture_ccm_distance_with_derived_via_cache(
                p,
                cfg,
                &alpha,
                &distance.rules,
                distance.profile_steps,
                &observed,
                id == "distance_resolution",
                id == "target_residual_analysis",
                id == "deviation_decomposition",
            )?;
            if id == "distance_resolution" {
                distance_resolution_verdicts = Some((
                    captured.resolution_tolerance_met,
                    captured.resolution_ladder_tolerance_met,
                ));
            }
        }
        let mut result = recorded(sink.finish()?)?;
        if let Some((reported, ladder)) = distance_resolution_verdicts {
            result = CapturedDiagnostic::new(
                &serde_json::json!({
                    "method": "returned_q_and_final_pair_resolution_verdicts_v1",
                    "claim_scope": "empirical_adjacent_grid_agreement_not_integral_error_bound",
                    "reported_resolution_tolerance_met": reported,
                    "refinement_ladder_tolerance_met": ladder,
                    "retained_evidence": result.value,
                }),
                result.sources,
            )?;
        }
        if id == "u_flow_response" {
            self.uflow_capture = Some(result.clone());
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn retained_run_survives_diagnostic_failure_and_authenticates_warm_sources() {
        retained_run_round_trip(false, false);
    }

    #[test]
    fn retained_run_captures_cold_artifacts_with_encoded_publication_staging() {
        retained_run_round_trip(true, false);
    }

    #[test]
    fn retained_run_observes_payload_when_reusing_a_larger_root_window() {
        retained_run_round_trip(false, true);
    }

    #[test]
    fn retained_roots_replay_adopted_published_validation_chain() {
        use xc_cache::*;
        let directory = xc_core::test_support::TestDir::new("retained-published-roots");
        let author = CacheResolver::new(vec![CacheLayer {
            precedence: 0,
            store: Box::new(FilesystemCacheStore::new(
                "local",
                directory.join("author"),
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
        let sink = CanonicalStagingProductionSink::new(
            directory.join("staging"),
            TransportPolicy::default(),
            xc_core::ResourcePolicy::default(),
            xc_core::CancellationToken::new(),
        )
        .unwrap();
        let context = ArtifactCacheContext {
            resolver: Some(&author),
            reference_resolver: None,
            acceptance: Some(&policy),
            ordered_overlays: vec!["local".into()],
            mode: ArtifactExecutionCacheMode::PreferReuse,
            write_on_miss: true,
            write_visibility: CacheVisibility::Local,
            requested_assurance: xc_core::AssuranceLevel::Computed,
            certification_failure_policy: CertificationFailurePolicy::RetainComputedFailRun,
            production_sink: Some(&sink),
        };
        let params = CcmParams::from_lambda_sq_integer(13, 16);
        let mut cfg = HighPrecConfig::for_decimal_digits(40).with_adaptive_root_precision();
        cfg.n_eigenvalues = 1;
        let dataset = xc_zeta::zeros::bundled_dataset_identity().unwrap();
        let seeds = vec![Float::with_val(
            cfg.precision_bits,
            Float::parse(&xc_zeta::zeros::bundled_first_n_strings(1).unwrap()[0]).unwrap(),
        )];
        let mut cold =
            RetainedCcmRun::seeded(&params, &cfg, 1, &seeds, &dataset, &context).unwrap();
        let options = super::super::super::capture::CcmCapturePlan::ultra(2, 17)
            .unwrap()
            .primary_options()
            .unwrap();
        let expected = cold
            .capture_diagnostic("root_band", &options, &context)
            .unwrap();
        let adopted_store = FilesystemCacheStore::new(
            "adopted",
            directory.join("adopted"),
            true,
            CacheVisibility::Local,
        );
        for draft in sink.drafts().unwrap() {
            let record = author
                .resolve_exact(
                    &draft.source_artifact_key,
                    &draft.source_content_digest,
                    CacheQuality::Validated,
                    &policy,
                )
                .unwrap();
            let mut tags = record.manifest.tags.clone();
            tags.insert(
                SEMANTIC_KEY_MANIFEST_TAG.into(),
                serde_json::to_string(&draft.manifest.semantic_key).unwrap(),
            );
            tags.insert(
                REMOTE_CANONICAL_MANIFEST_TAG.into(),
                serde_json::to_string(&draft.manifest).unwrap(),
            );
            // Match the shard adapter's authenticated canonical metadata,
            // including its intentionally empty local dependency list.
            adopted_store
                .put(
                    &ArtifactDraft {
                        schema_version: 1,
                        key: record.manifest.key,
                        producer_toolkit_version: record.manifest.producer_toolkit_version,
                        minimum_reader_version: record.manifest.minimum_reader_version,
                        maximum_reader_version: record.manifest.maximum_reader_version,
                        quality: record.manifest.quality,
                        visibility: CacheVisibility::Local,
                        immutable: true,
                        dependencies: vec![],
                        tags,
                        provenance_digest: Some(draft.manifest.digest().unwrap()),
                    },
                    &record.payload,
                )
                .unwrap();
        }
        let adopted = CacheResolver::new(vec![CacheLayer {
            precedence: 0,
            store: Box::new(adopted_store),
        }]);
        let context = ArtifactCacheContext {
            resolver: Some(&adopted),
            ordered_overlays: vec!["adopted".into()],
            mode: ArtifactExecutionCacheMode::RequireReuse,
            write_on_miss: false,
            production_sink: None,
            ..context
        };
        let mut warm =
            RetainedCcmRun::seeded(&params, &cfg, 1, &seeds, &dataset, &context).unwrap();
        assert_eq!(
            cold.primary().eigenvalues_pos[0].value(),
            warm.primary().eigenvalues_pos[0].value()
        );
        for manifest in [&warm.source.root_manifest, &warm.source.secular_manifest] {
            let manifest = manifest.as_ref().unwrap();
            assert!(manifest.dependencies.is_empty());
            assert!(retained_canonical_manifest(manifest).unwrap().is_some());
        }
        let actual = warm
            .capture_diagnostic("root_band", &options, &context)
            .unwrap();
        assert_eq!(actual.value, expected.value);
    }

    #[cfg(feature = "arb")]
    #[test]
    fn sector_certificate_is_fixed_by_the_first_request_and_bound_to_its_manifest() {
        use crate::ccm::sector_gap_certificate::CcmSectorGapCertificationOptions;
        use xc_cache::{
            ArtifactExecutionCacheMode, CacheLayer, CachePolicy, CacheResolver, CacheVisibility,
            ZipJsonFilesystemCacheStore as FilesystemCacheStore,
        };
        let root_dir = xc_core::test_support::TestDir::new("ccm-retained-certificate");
        let resolver = CacheResolver::new(vec![CacheLayer {
            precedence: 0,
            store: Box::new(FilesystemCacheStore::new(
                "local",
                root_dir.to_path_buf(),
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
        let cache = ArtifactCacheContext {
            resolver: Some(&resolver),
            reference_resolver: None,
            acceptance: Some(&policy),
            ordered_overlays: vec!["local".into()],
            mode: ArtifactExecutionCacheMode::PreferReuse,
            write_on_miss: true,
            write_visibility: CacheVisibility::Local,
            requested_assurance: xc_core::AssuranceLevel::Computed,
            certification_failure_policy:
                xc_cache::CertificationFailurePolicy::RetainComputedFailRun,
            production_sink: None,
        };
        let params = CcmParams::from_lambda_sq_integer(13, 16);
        let mut cfg = HighPrecConfig::for_decimal_digits(40);
        cfg.n_eigenvalues = 1;
        let dataset = xc_zeta::zeros::bundled_dataset_identity().unwrap();
        let seeds = vec![Float::with_val(
            cfg.precision_bits,
            Float::parse(&xc_zeta::zeros::bundled_first_n_strings(1).unwrap()[0]).unwrap(),
        )];
        let mut run = RetainedCcmRun::seeded(&params, &cfg, 1, &seeds, &dataset, &cache).unwrap();
        let mut options = super::super::super::capture::CcmCapturePlan::ultra(2, 17)
            .unwrap()
            .primary_options()
            .unwrap();
        options.sector_gap_certification = Some(CcmSectorGapCertificationOptions::default());
        let first = run
            .capture_diagnostic("sector_gap_certificate", &options, &cache)
            .unwrap();
        assert!(first.value_reference.is_some(), "recorded by reference");
        let again = run
            .capture_diagnostic("sector_gap_certificate", &options, &cache)
            .unwrap();
        assert_eq!(first.value, again.value);

        let mut finer = options.clone();
        finer.sector_gap_certification = Some(CcmSectorGapCertificationOptions {
            relative_enclosure_bits: 24,
            ..CcmSectorGapCertificationOptions::default()
        });
        let Err(error) = run.capture_diagnostic("sector_gap_certificate", &finer, &cache) else {
            panic!("a changed request must not reuse the certificate");
        };
        assert!(
            format!("{error:#}").contains("certification options changed"),
            "{error:#}"
        );
        let mut wider = options.clone();
        wider.sector_analysis = options
            .sector_analysis
            .map(|sector| CcmSectorAnalysisOptions {
                requested_eigenpairs: sector.requested_eigenpairs + 1,
                ..sector
            });
        let Err(error) = run.capture_diagnostic("sector_gap_certificate", &wider, &cache) else {
            panic!("a changed request must not reuse the certificate");
        };
        assert!(
            format!("{error:#}").contains("sector analysis options changed"),
            "{error:#}"
        );

        let resolved = run.sector_certificate.as_ref().unwrap();
        let manifest = resolved
            .produced_manifest
            .as_ref()
            .or(resolved.reused_manifest.as_ref())
            .unwrap();
        assert!(xc_cache::json_payload_matches_manifest(manifest, &resolved.value).unwrap());
        let mut altered = resolved.value.clone();
        altered.certifies_finite_ground_state_simple =
            !altered.certifies_finite_ground_state_simple;
        assert!(!xc_cache::json_payload_matches_manifest(manifest, &altered).unwrap());

        // Source-only acquisition survives an unrelated optional-input failure,
        // while retaining both that failure and the actual certificate error.
        run.extended_inputs_loaded = true;
        run.extended_input_error = Some("synthetic optional preparation failure".into());
        run.sector_certificate = None;
        run.sector_certificate_error = Some("synthetic inertia did not separate".into());
        let plan = crate::ccm::capture::CcmCapturePlan::ultra(2, 17).unwrap();
        let ids = [
            "finite_root_budget",
            "trial_vector_energy",
            "trial_vector_parity",
            "indexed_prolate_comparison",
        ];
        let result = xc_cache::capture_and_persist(
            &plan,
            ids.iter().map(|s| s.to_string()).collect(),
            |id| run.capture_diagnostic_outcome(id, &options, &cache),
            &cache,
        )
        .unwrap();
        for id in ids {
            let value =
                xc_cache::measurement_value(&result.value.measurements[id], &resolver, &policy)
                    .unwrap();
            if id == "finite_root_budget" {
                assert_eq!(value["data"]["outcome"], "blocked");
                assert!(value["data"]["reason"]
                    .as_str()
                    .unwrap()
                    .contains("synthetic inertia did not separate"));
            } else {
                assert_eq!(value["data"]["outcome"], "computed");
                assert_eq!(
                    value["data"]["input_preparation_error"],
                    "synthetic optional preparation failure"
                );
            }
        }
        assert!(run
            .capture_diagnostic("constrained_l1_fit", &options, &cache)
            .is_err());
    }

    fn retained_run_round_trip(with_staging: bool, wider_window: bool) {
        use xc_cache::{
            ArtifactExecutionCacheMode, CacheLayer, CachePolicy, CacheResolver, CacheVisibility,
            ZipJsonFilesystemCacheStore as FilesystemCacheStore,
        };
        let root_dir = xc_core::test_support::TestDir::new("ccm-retained-run");
        let root = root_dir.to_path_buf();
        let resolver = CacheResolver::new(vec![CacheLayer {
            precedence: 0,
            store: Box::new(FilesystemCacheStore::new(
                "local",
                &root,
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
        let staging = with_staging.then(|| {
            xc_cache::CanonicalStagingProductionSink::new(
                root.join("publication"),
                xc_cache::TransportPolicy::default(),
                xc_core::ResourcePolicy::default(),
                xc_core::CancellationToken::new(),
            )
            .unwrap()
        });
        if let Some(sink) = &staging {
            assert!(sink.supports_encoded_records());
        }
        let cache = ArtifactCacheContext {
            resolver: Some(&resolver),
            reference_resolver: None,
            acceptance: Some(&policy),
            ordered_overlays: vec!["local".into()],
            mode: ArtifactExecutionCacheMode::PreferReuse,
            write_on_miss: true,
            write_visibility: CacheVisibility::Local,
            requested_assurance: xc_core::AssuranceLevel::Computed,
            certification_failure_policy:
                xc_cache::CertificationFailurePolicy::RetainComputedFailRun,
            production_sink: staging
                .as_ref()
                .map(|sink| sink as &dyn ArtifactProductionSink),
        };
        let params = CcmParams::from_lambda_sq_integer(13, if wider_window { 120 } else { 16 });
        let mut cfg = HighPrecConfig::for_decimal_digits(40);
        cfg.n_eigenvalues = 1;
        let dataset = xc_zeta::zeros::bundled_dataset_identity().unwrap();
        let strings = xc_zeta::zeros::bundled_first_n_strings(1).unwrap();
        let seeds = vec![Float::with_val(
            cfg.precision_bits,
            Float::parse(&strings[0]).unwrap(),
        )];
        let wider = if wider_window {
            let wider_seeds = xc_zeta::zeros::bundled_first_n_strings(25)
                .unwrap()
                .iter()
                .map(|s| Float::with_val(cfg.precision_bits, Float::parse(s).unwrap()))
                .collect::<Vec<_>>();
            let mut wider_cfg = cfg.clone();
            wider_cfg.n_eigenvalues = 25;
            Some(
                RetainedCcmRun::seeded(&params, &wider_cfg, 1, &wider_seeds, &dataset, &cache)
                    .unwrap(),
            )
        } else {
            None
        };
        let mut run = RetainedCcmRun::seeded(&params, &cfg, 1, &seeds, &dataset, &cache).unwrap();
        if let Some(wider) = &wider {
            let manifest = run.source.root_manifest.as_ref().unwrap();
            assert_eq!(
                manifest.content_digest,
                wider.source.root_manifest.as_ref().unwrap().content_digest
            );
            assert!(run
                .retained_records
                .iter()
                .any(|r| r.manifest.key == manifest.key
                    && r.manifest.content_digest == manifest.content_digest));
            assert_eq!(run.primary.eigenvalues_pos.len(), 1);
            assert_eq!(
                run.primary.eigenvalues_pos[0].value(),
                wider.primary.eigenvalues_pos[0].value()
            );
        }
        let initial = PortableHighPrecResult::from_runtime(run.primary()).unwrap();
        let mut options = super::super::super::capture::CcmCapturePlan::ultra(2, 17)
            .unwrap()
            .primary_options()
            .unwrap();
        options.distance_capture = Some(CcmDistanceCaptureOptions::default_convention(32, 32));
        if wider_window {
            for id in ["indexed_transform", "root_band"] {
                let captured = run.capture_diagnostic(id, &options, &cache).unwrap();
                assert_eq!(
                    captured.value["data"][if id == "root_band" { "points" } else { "rows" }]
                        .as_array()
                        .unwrap()
                        .len(),
                    25
                );
            }
            assert_eq!(
                serde_json::to_value(initial).unwrap(),
                serde_json::to_value(PortableHighPrecResult::from_runtime(run.primary()).unwrap())
                    .unwrap()
            );
            std::fs::remove_dir_all(root).unwrap();
            return;
        }
        if with_staging {
            // Fresh state/matrix notifications must survive encoded production too.
            assert_eq!(run.retained_even_sources(&cache).unwrap().1.len(), 1);
            options.root_certification = Some(
                CcmRootCertificationOptions::for_decimal_digits(
                    crate::ccm::certified_roots::IndependentCcmRootTarget::Prefix { count: 3 },
                    20,
                )
                .unwrap(),
            );
            let mut ids = vec![
                "checkpoint_spectra",
                "root_certificate",
                "target_comparison",
            ];
            if cfg!(feature = "arb") {
                ids.push("assembly_error");
            }
            for id in ids {
                let cold = run
                    .capture_diagnostic(id, &options, &cache)
                    .unwrap_or_else(|error| panic!("cold {id}: {error:#}"));
                let warm = run
                    .capture_diagnostic(id, &options, &cache)
                    .unwrap_or_else(|error| panic!("warm {id}: {error:#}"));
                assert_eq!(cold.value, warm.value, "{id}");
                assert!(
                    cold.value_reference.is_some() || id == "root_certificate",
                    "{id}"
                );
                match id {
                    "root_certificate" => {
                        let rows = cold.value["rows"].as_array().unwrap();
                        assert_eq!(rows.len(), 3);
                        assert_eq!(rows[0]["computed_status"], "converged");
                        assert!(rows
                            .iter()
                            .all(|row| row["outcome"] != "not_computed_not_certified"
                                || row["reason"].is_string()));
                    }
                    "target_comparison" => {
                        assert_eq!(cold.value["outcome"], "computed", "{}", cold.value);
                        assert_eq!(cold.value["levels"].as_array().unwrap().len(), 2);
                        let residual: f64 = cold.value["basis_consistency_residual"]
                            .as_str()
                            .unwrap()
                            .parse()
                            .unwrap();
                        assert!(residual < 1e-30, "{residual}");
                        assert!(
                            cold.value["levels"][1]["projection"]["rayleigh_quotient"].is_string()
                        );
                        assert!(!xc_cache::artifact_kind_admitted_to_destination(
                            crate::distance::hp::TARGET_COMPARISON_KIND,
                            xc_cache::PublicationDestination::Public
                        ));
                    }
                    "assembly_error" => {
                        assert_eq!(cold.value["outcome"], "certified_finite_enclosure");
                        assert!(!cold.value["exact_form_bounds"]
                            .as_array()
                            .unwrap()
                            .is_empty());
                    }
                    _ => assert!(!cold.value["rows"].as_array().unwrap().is_empty()),
                }
            }
            for id in [
                "capture_preflight",
                "consistency",
                "configuration_comparison",
                "band_reconstruction",
                "transform_enclosure",
                "arithmetic_energy_full",
                "directional_response_full",
                "spectral_cluster_full",
                "complex_transform",
                "root_transport",
                "operator_cluster",
                "finite_section_transfer",
                "observable_budget",
                "state_geometry",
                "indexed_transform",
                "operator_energy",
                "root_band",
                "deviation_decomposition",
                "distance_resolution",
                "evenness",
                "prime_power_response",
                "target_residual_analysis",
                "u_flow_response",
                "sector_analysis",
                "target_distance",
            ] {
                let cold = run
                    .capture_diagnostic(id, &options, &cache)
                    .unwrap_or_else(|error| panic!("cold {id}: {error:#}"));
                let warm = run
                    .capture_diagnostic(id, &options, &cache)
                    .unwrap_or_else(|error| panic!("warm {id}: {error:#}"));
                assert_eq!(cold.value, warm.value, "{id}");
                assert_eq!(
                    cold.sources
                        .iter()
                        .map(|m| &m.content_digest)
                        .collect::<Vec<_>>(),
                    warm.sources
                        .iter()
                        .map(|m| &m.content_digest)
                        .collect::<Vec<_>>(),
                    "{id}"
                );
            }
            assert!(run
                .retained_trial_components(128, 1, &cache)
                .unwrap()
                .is_none());
            let retained_components = run
                .retained_trial_components(128, 48 * 1024 * 1024, &cache)
                .unwrap()
                .unwrap();
            assert_eq!(retained_components.components.len(), 3);
            assert_eq!(retained_components.components[2].data.signed_weight, "-1");
            assert!(run
                .prepare_retained_trial_components(128, 48 * 1024 * 1024, &cache)
                .unwrap());
            for kind in ["ccm_archimedean_integrals", "ccm_prime_component"] {
                assert!(run
                    .extended_research_sources()
                    .iter()
                    .any(|m| m.key.kind == kind));
            }
            {
                use crate::ccm::convergence_capture::finite_capture as f;
                let live = run
                    .capture_diagnostic("trial_vector_energy", &options, &cache)
                    .unwrap();
                let replay =
                    f::RetainedFiniteSources::from_manifests(&run.primary_sources(), true, &cache)
                        .unwrap()
                        .capture(
                            "trial_vector_energy",
                            run.extended_inputs.as_ref(),
                            &run.run_once_sources,
                            &cache,
                        )
                        .unwrap();
                assert_eq!(live.value, serde_json::to_value(replay.value).unwrap());
                assert!(replay.reused_manifest.is_some());
                let components = &live.value["data"]["result"]["component_energy"];
                assert_eq!(components["components"].as_array().unwrap().len(), 3);
                assert_ne!(components["operator_closure"]["status"], "refuted");
                let mut ancestry = Vec::new();
                for manifest in &live.sources {
                    ancestry.extend(
                        xc_cache::resolve_manifest_sources(manifest, &run.run_once_sources, &cache)
                            .unwrap(),
                    );
                }
                let first_level = ancestry.clone();
                for manifest in &first_level {
                    ancestry.extend(
                        xc_cache::resolve_manifest_sources(manifest, &run.run_once_sources, &cache)
                            .unwrap(),
                    );
                }
                for kind in ["ccm_archimedean_integrals", "ccm_prime_component"] {
                    assert!(
                        ancestry.iter().any(|m| m.key.kind == kind),
                        "retained component parent {kind}"
                    );
                }
            }
            let inputs = run.extended_inputs.as_ref().unwrap();
            let auto = inputs.run_once.as_ref().unwrap();
            assert_eq!(auto.component_actions.len(), 3, "{:?}", auto.producer_notes);
            assert!(
                !auto.derivative_actions.is_empty(),
                "{:?}",
                auto.producer_notes
            );
            assert_eq!(inputs.cluster.len(), 4, "{:?}", auto.producer_notes);
            assert!(run.uflow_capture.is_some());
            // All new automatic measurements share the existing Ultra groups.
            // Acquire first, then bind a synthetic private provider and prove
            // that retained-only replay produces the identical artifacts.
            {
                use crate::ccm::convergence_capture::finite_capture as f;
                use crate::ccm::extended_research::SampledReference;
                let mut supplied = inputs.clone();
                let mut coefficients = vec!["0".to_owned(); params.matrix_size()];
                coefficients[params.n_modes] = "1".into();
                supplied.target = Some(SampledReference {
                    definition_digest: ContentDigest::sha256(b"synthetic constant target"),
                    evaluation_policy: "stored synthetic constant".into(),
                    approximation_scope: "finite profile only".into(),
                    intervals: 8,
                    values: vec!["1".into(); 9],
                    basis_values: vec![],
                    fixed_second_component: None,
                    raw_normalizer: "1".into(),
                    trial_coefficients: Some(coefficients.clone()),
                });
                let baseline = f::ComplexVector {
                    label: "synthetic trial".into(),
                    real: coefficients
                        .iter()
                        .map(xc_solver::trial_energy::ExactBounds::point)
                        .collect(),
                    imaginary: vec![
                        xc_solver::trial_energy::ExactBounds::point("0");
                        params.matrix_size()
                    ],
                };
                supplied.finite_diagnostics = Some(f::Inputs {
                    scope: "synthetic captured complex input".into(),
                    complex_trials: Some(f::ComplexTrialSeries {
                        basis_id: "centered_full_V_fourier".into(),
                        matrix_digest: run
                            .source
                            .tau_manifest
                            .as_ref()
                            .unwrap()
                            .content_digest
                            .clone(),
                        scope: "synthetic finite trial".into(),
                        baseline,
                        corrections: vec![],
                        functional: None,
                        provenance: BTreeMap::new(),
                    }),
                    ..Default::default()
                });
                {
                    use f::energy_extensions as e;
                    use xc_solver::trial_energy::ExactBounds;
                    let extra = supplied.finite_diagnostics.as_mut().unwrap();
                    let matrix_digest = run
                        .source
                        .tau_manifest
                        .as_ref()
                        .unwrap()
                        .content_digest
                        .clone();
                    let n = params.matrix_size();
                    let mut upper_triangle = vec![];
                    for i in 0..n {
                        for j in i..n {
                            let value: rug::Rational =
                                (run.source.tau[i * n + j].to_rational().unwrap()
                                    + run.source.tau[j * n + i].to_rational().unwrap())
                                    / 2;
                            upper_triangle.push(ExactBounds::point(value.to_string()));
                        }
                    }
                    extra.component_energy = Some(e::ComponentRequest {
                        matrix_digest,
                        basis_id: "centered_full_V_fourier".into(),
                        components: vec![e::SignedComponent::from_data(e::ComponentData {
                            label: "stored form".into(),
                            signed_weight: "1".into(),
                            diagonal: vec![],
                            upper_triangle,
                            rank_one: vec![],
                        })
                        .unwrap()],
                        assembly_operator_norm_error: None,
                    });
                    let zero = e::DeclaredBound {
                        upper: "0".into(),
                        provenance: "exact synthetic polynomial".into(),
                        scope: "specified polynomial".into(),
                    };
                    extra.projection_energy = Some(e::ProjectionRequest {
                        definition_digest: ContentDigest::sha256(b"synthetic projection"),
                        operator_id: "separate synthetic interval form".into(),
                        period: "2".into(),
                        basis_id: "orthonormal_periodic_fourier".into(),
                        domain: "interval_h1".into(),
                        coefficients: extra.complex_trials.as_ref().unwrap().baseline.clone(),
                        retained_modes: 0,
                        remainder: e::FourierRemainder::Unknown,
                        source_l2_error: zero.clone(),
                        source_h1_error: zero,
                        continuity: None,
                        trial_absolute_energy: None,
                        precision_bits: 128,
                    });
                    let triangle = e::LogProfile {
                        label: "synthetic compact tent".into(),
                        knots: vec![
                            e::Knot {
                                coordinate: "-1/4".into(),
                                value: "0".into(),
                            },
                            e::Knot {
                                coordinate: "0".into(),
                                value: "1".into(),
                            },
                            e::Knot {
                                coordinate: "1/4".into(),
                                value: "0".into(),
                            },
                        ],
                    };
                    extra.continuous_energy = Some(e::ContinuousRequest {
                        definition_digest: ContentDigest::sha256(b"synthetic continuous form"),
                        profile_a: triangle.clone(),
                        profile_b: triangle.clone(),
                        profile_sum: triangle,
                        scope: e::ProfileScope::UnresolvedSource,
                        summands: vec![],
                        integration_windows: vec![],
                        unrepresented_sum_l2: None,
                        overlap: None,
                        translation_shifts: vec![],
                        integration_cells: 16,
                        maximum_prime_power: 8,
                        precision_bits: 128,
                    });
                }
                run.set_extended_research_inputs(supplied).unwrap();
                for id in [
                    "trial_vector_energy",
                    "trial_vector_parity",
                    "normalization_error_bound",
                    "continuous_l1_bound",
                    "finite_tail_bound",
                ] {
                    let live = run
                        .capture_diagnostic(id, &options, &cache)
                        .unwrap_or_else(|e| panic!("live {id}: {e:#}"));
                    let replay = f::RetainedFiniteSources::from_manifests(
                        &run.primary_sources(),
                        true,
                        &cache,
                    )
                    .unwrap()
                    .capture(
                        id,
                        run.extended_inputs.as_ref(),
                        &run.run_once_sources,
                        &cache,
                    )
                    .unwrap();
                    assert_eq!(
                        live.value,
                        serde_json::to_value(replay.value).unwrap(),
                        "retained-only {id}"
                    );
                    assert!(replay.reused_manifest.is_some(), "warm replay {id}");
                    let data = &live.value["data"]["result"];
                    match id {
                        "trial_vector_energy" => {
                            assert_eq!(
                                data["component_energy"]["operator_closure"]["status"],
                                "exact_stored_equality"
                            );
                            assert_eq!(
                                data["continuous_energy"]["source_global_status"],
                                "unresolved"
                            );
                            assert!(data["functional_energy"]["rows"].is_array());
                        }
                        "finite_tail_bound" => {
                            assert_eq!(data["projection_energy"]["status"], "unresolved")
                        }
                        "continuous_l1_bound" => assert!(data["paired_profiles"]["pairs"]
                            .as_array()
                            .is_some_and(|p| p.len() == 6)),
                        "normalization_error_bound" => assert!(data["paired_normalization"]
                            ["functional_operator_norm_bound"]
                            .is_object()),
                        _ => assert!(data["functional_energy"]["rows"].is_array()),
                    }
                }
                let retained =
                    f::RetainedFiniteSources::from_manifests(&run.primary_sources(), true, &cache)
                        .unwrap();
                let observation = retained
                    .refinement_observation(
                        f::RefinementConfiguration {
                            observable: f::RefinementObservable::Eigenvalue,
                            domain: f::ComparisonDomain::CoefficientSpace,
                            lambda_squared: "13".into(),
                            branch: f::SpectralBranch::EvenGround,
                            external_index: 0,
                            index_origin: f::IndexOrigin::Zero,
                            n_modes: params.n_modes,
                            precision_bits: cfg.precision_bits,
                            operator_quadrature_orders: cfg
                                .resolved_archimedean_orders(
                                    params.n_modes,
                                    &log_lambda_sq_hp(&params, cfg.precision_bits).unwrap(),
                                )
                                .unwrap(),
                            projection_quadrature_order: 32,
                            representation_order: params.n_modes,
                            guard_bits: 64,
                            certificate_precision_bits: cfg.precision_bits,
                            basis_id: "centered_full_V_fourier".into(),
                            metric_id: "identity-coefficient-metric-v1".into(),
                            operator_recipe: ContentDigest::sha256(
                                b"synthetic qualification recipe",
                            ),
                            target_definition: None,
                            normalizer_id: "unit coefficient norm".into(),
                            trial_recipe: None,
                        },
                        f::ObservationValidity::Accepted,
                    )
                    .unwrap();
                let mut cohort = f::RefinementCohort {
                    axis: f::RefinementAxis::Dimension,
                    scope: "one retained point; branch label is not a certificate".into(),
                    relative_tolerance: "0.01".into(),
                    observations: vec![observation],
                };
                let report =
                    f::capture_retained_refinement_cohort(&cohort, &[&retained], &cache).unwrap();
                assert_eq!(report.value.data["operationally_stabilized"], false);
                assert_eq!(report.value.source_dependencies.len(), 2);
                assert!(report.value.data["source_authentication"].is_string());
                cohort.observations[0].value = "0".into();
                assert!(
                    f::capture_retained_refinement_cohort(&cohort, &[&retained], &cache).is_err()
                );
            }
            let inputs = run.extended_inputs.as_ref().unwrap();
            let mut foreign = inputs.clone();
            foreign.source_eigenpair = ContentDigest::sha256(b"foreign state");
            assert!(run.set_extended_research_inputs(foreign.clone()).is_err());
            // Simulate a structurally valid but foreign file loaded by the environment adapter.
            run.extended_inputs = Some(foreign);
            run.run_once_prepared = false;
            assert!(run
                .capture_diagnostic("complex_transform", &options, &cache)
                .is_ok());
            assert!(run
                .capture_diagnostic("signed_transform", &options, &cache)
                .is_err());
            run.extended_input_error = None;

            assert!(!staging.as_ref().unwrap().drafts().unwrap().is_empty());
        }
        assert!(run.capture_diagnostic("unknown", &options, &cache).is_err());
        let first = run
            .capture_diagnostic("root_conditioning", &options, &cache)
            .unwrap();
        let second = run
            .capture_diagnostic("root_conditioning", &options, &cache)
            .unwrap();
        assert_eq!(first.value, second.value);
        assert_eq!(
            first.sources[0].content_digest,
            second.sources[0].content_digest
        );
        let profile = run
            .capture_diagnostic("distance_profile", &options, &cache)
            .unwrap();
        assert_eq!(profile.sources[0].key.kind, "ccm_eigenfunction_profile");
        let sink = RecordingSink::new(&cache, &["ccm_eigenfunction_profile"]);
        let mut refresh = observing(&cache, &sink);
        refresh.mode = ArtifactExecutionCacheMode::Refresh;
        let distance = options.distance_capture.as_ref().unwrap();
        crate::distance::hp::capture_ccm_distance_via_cache(
            &params,
            &cfg,
            &Float::with_val(cfg.precision_bits, 1),
            &distance.rules,
            distance.profile_steps,
            &refresh,
        )
        .unwrap();
        let original_route = sink.finish().unwrap();
        assert_eq!(
            profile.sources[0].content_digest,
            original_route[0].manifest.content_digest
        );
        let (matrix, eigenpairs) = run.retained_even_sources(&cache).unwrap();
        assert_eq!(matrix.dimension(), 17);
        assert_eq!(eigenpairs.len(), 1);
        assert_eq!(
            serde_json::to_value(initial).unwrap(),
            serde_json::to_value(PortableHighPrecResult::from_runtime(run.primary()).unwrap())
                .unwrap()
        );
        let mut natural_cfg = cfg.clone();
        natural_cfg.set_parity_policy(CcmParityPolicy::Natural);
        let mut natural =
            RetainedCcmRun::seeded(&params, &natural_cfg, 1, &seeds, &dataset, &cache).unwrap();
        assert!(natural
            .capture_diagnostic("prime_power_response", &options, &cache)
            .is_err());
        assert!(natural.retained_even_sources(&cache).unwrap().1.is_empty());
        let geometry = natural
            .capture_diagnostic("state_geometry", &options, &cache)
            .unwrap();
        assert_eq!(geometry.sources[0].key.kind, "ccm_state_geometry_analysis");
        assert_eq!(
            geometry.sources[0].dependencies[0].content_digest,
            natural.eigenpair.as_ref().unwrap().manifest.content_digest
        );
        let original_eigenpair = run.eigenpair.take().unwrap();
        assert!(run
            .capture_diagnostic("state_geometry", &options, &cache)
            .is_err());
        run.eigenpair = Some(original_eigenpair);
        assert!(run
            .capture_diagnostic("state_geometry", &options, &cache)
            .is_ok());

        std::fs::remove_dir_all(root).unwrap();
    }

    // Every file below `root` as (relative path, content). Only creation
    // times may differ between runs: they are removed, with every manifest
    // digest (which covers its creation time) and the file and staging-draft
    // names derived from those digests.
    fn cache_snapshot(root: &std::path::Path) -> Vec<(String, String)> {
        fn untimed(text: &str) -> String {
            let mut out = String::new();
            let mut rest = text;
            while let Some(at) = rest.find("manifests/") {
                let (head, tail) = rest.split_at(at + "manifests/".len());
                out.push_str(head);
                let name = tail.split(['"', '\\']).next().unwrap_or_default();
                let timed = name.strip_suffix(".json").and_then(|n| n.split_once('-'));
                if timed.is_some_and(|(time, digest)| {
                    time.bytes().all(|b| b.is_ascii_digit()) && digest.len() == 64
                }) {
                    out.push_str("<manifest>.json");
                    rest = &tail[name.len()..];
                } else {
                    rest = tail;
                }
            }
            out.push_str(rest);
            out
        }
        fn strip(value: &mut serde_json::Value) {
            match value {
                serde_json::Value::Object(map) => {
                    // A staged path names its run directory and source manifest.
                    map.retain(|key, _| {
                        !matches!(key.as_str(), "created_unix_seconds" | "staged_parts_root")
                            && !key.ends_with("manifest_digest")
                    });
                    map.values_mut().for_each(strip);
                }
                serde_json::Value::Array(items) => items.iter_mut().for_each(strip),
                _ => {}
            }
        }
        let mut files = Vec::new();
        let mut pending = vec![root.to_path_buf()];
        while let Some(directory) = pending.pop() {
            for entry in std::fs::read_dir(directory).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    pending.push(path);
                    continue;
                }
                let bytes = std::fs::read(&path).unwrap();
                let content = match serde_json::from_slice::<serde_json::Value>(&bytes) {
                    Ok(mut value) => {
                        strip(&mut value);
                        untimed(&value.to_string())
                    }
                    Err(_) => ContentDigest::sha256(&bytes).0,
                };
                let mut name = path.strip_prefix(root).unwrap().display().to_string();
                if name.starts_with("publication/drafts/") {
                    // drafts/<semantic>/<payload>/<source manifest digest>/...
                    let mut parts = name.split('/').map(str::to_owned).collect::<Vec<_>>();
                    if parts.len() > 4 {
                        parts[4] = "<source-manifest>".into();
                    }
                    name = parts.join("/");
                }
                files.push((untimed(&name), content));
            }
        }
        files.sort();
        files
    }

    /// One cold capture through the receipt collector, as an application
    /// drives it: the persisted record, every measurement value and outcome,
    /// and all cache and publication-staging files. Also returns how many
    /// results came from the look-ahead lane. `wait` lets the lane finish after
    /// each capture, so every look-ahead result is taken; otherwise claims
    /// race the lane.
    fn lookahead_sequence(
        lookahead: bool,
        wait: bool,
        failing_trials: bool,
    ) -> (Vec<(String, String)>, usize) {
        use xc_cache::{
            ArtifactExecutionCacheMode, CacheLayer, CachePolicy, CacheResolver, CacheVisibility,
            ZipJsonFilesystemCacheStore as FilesystemCacheStore,
        };
        let root_dir = xc_core::test_support::TestDir::new("ccm-lookahead");
        let root = root_dir.to_path_buf();
        let resolver = CacheResolver::new(vec![CacheLayer {
            precedence: 0,
            store: Box::new(FilesystemCacheStore::new(
                "local",
                root.join("cache"),
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
        let staging = xc_cache::CanonicalStagingProductionSink::new(
            root.join("publication"),
            xc_cache::TransportPolicy::default(),
            xc_core::ResourcePolicy::default(),
            xc_core::CancellationToken::new(),
        )
        .unwrap();
        let cache = ArtifactCacheContext {
            resolver: Some(&resolver),
            reference_resolver: None,
            acceptance: Some(&policy),
            ordered_overlays: vec!["local".into()],
            mode: ArtifactExecutionCacheMode::PreferReuse,
            write_on_miss: true,
            write_visibility: CacheVisibility::Local,
            requested_assurance: xc_core::AssuranceLevel::Computed,
            certification_failure_policy:
                xc_cache::CertificationFailurePolicy::RetainComputedFailRun,
            production_sink: Some(&staging),
        };
        let params = CcmParams::from_lambda_sq_integer(13, 16);
        let mut cfg = HighPrecConfig::for_decimal_digits(40);
        cfg.n_eigenvalues = 1;
        let dataset = xc_zeta::zeros::bundled_dataset_identity().unwrap();
        let strings = xc_zeta::zeros::bundled_first_n_strings(1).unwrap();
        let seeds = vec![Float::with_val(
            cfg.precision_bits,
            Float::parse(&strings[0]).unwrap(),
        )];
        let mut run = RetainedCcmRun::seeded(&params, &cfg, 1, &seeds, &dataset, &cache).unwrap();
        run.set_capture_lookahead(lookahead);
        run.set_lookahead_requests(
            crate::ccm::convergence_capture::lookahead::FINITE
                .iter()
                .chain(crate::ccm::convergence_capture::lookahead::EXTENDED)
                .map(|id| id.to_string()),
        );
        // Explicit inputs stand in for run-once preparation: its cohort
        // discovery reads registrations that concurrently running tests add
        // to the shared cache root, which would make two runs differ.
        use crate::ccm::convergence_capture::finite_capture as f;
        let record = run.eigenpair.as_ref().unwrap();
        let state = crate::ccm::state_geometry::RetainedState::from_payload(
            &record.manifest,
            &record.payload,
            std::slice::from_ref(&record.manifest.content_digest),
        )
        .unwrap();
        let mut input: crate::ccm::extended_research::ExternalResearchInputs =
            serde_json::from_value(serde_json::json!({
                "schema_version": 1,
                "source_eigenpair": record.manifest.content_digest,
                "lambda_squared": state.cutoff,
                "n_modes": state.modes,
                "precision_bits": state.precision,
                "convention_id": "synthetic_lookahead_inputs",
                "definition_digest": ContentDigest::sha256(b"synthetic look-ahead inputs"),
                "approximation_scope": "synthetic"
            }))
            .unwrap();
        if failing_trials {
            // Injected failure inside the look-ahead computation: a trial
            // vector of the wrong length passes input validation and fails
            // only when the trial-vector diagnostics compute.
            input.finite_diagnostics = Some(f::Inputs {
                scope: "synthetic trial of the wrong length".into(),
                trials: Some(f::TrialSeries {
                    basis_id: "centered_full_V_fourier".into(),
                    scope: "synthetic".into(),
                    baseline: f::Vector {
                        label: "short".into(),
                        coefficients: vec![xc_solver::trial_energy::ExactBounds::point("1")],
                    },
                    corrections: vec![],
                    provenance: BTreeMap::new(),
                }),
                ..Default::default()
            });
        }
        run.set_extended_research_inputs(input).unwrap();
        run.run_once_prepared = true;
        let plan = crate::ccm::capture::CcmCapturePlan::ultra(2, 17).unwrap();
        let options = plan.primary_options().unwrap();
        let ids = [
            "constrained_l1_fit",
            "continuous_l1_bound",
            "finite_section_transfer",
            "finite_tail_bound",
            "normalization_error_bound",
            "operator_cluster",
            "spectral_cluster_bound",
            "trial_vector_energy",
            "trial_vector_parity",
        ];
        let taken = crate::ccm::capture_runtime::lookahead_results_taken();
        let record = xc_cache::capture_and_persist(
            &plan,
            ids.iter().map(|id| id.to_string()).collect(),
            |id| {
                let outcome = run.capture_diagnostic_outcome(id, &options, &cache);
                if wait {
                    if let Some(lane) = &run.lookahead {
                        lane.wait_idle();
                    }
                }
                outcome
            },
            &cache,
        )
        .unwrap();
        let taken = crate::ccm::capture_runtime::lookahead_results_taken() - taken;
        assert_eq!(run.lookahead.is_some(), lookahead);
        drop(run);
        let mut outputs = vec![(
            "record".to_owned(),
            serde_json::to_string(&record.value).unwrap(),
        )];
        for (id, measurement) in &record.value.measurements {
            let value = xc_cache::measurement_value(measurement, &resolver, &policy).unwrap();
            outputs.push((id.clone(), value.to_string()));
        }
        for (id, outcome) in record.value.receipt.outcomes() {
            outputs.push((
                format!("outcome {id}"),
                serde_json::to_string(outcome).unwrap(),
            ));
        }
        outputs.extend(cache_snapshot(&root));
        (outputs, taken)
    }

    #[test]
    fn lookahead_runs_only_declared_requests_and_never_checkpointing_diagnostics() {
        use std::collections::BTreeSet;
        let reached = BTreeSet::new();
        assert!(lookahead_ids(None, &reached).is_empty());
        let only_energy = BTreeSet::from(["arithmetic_energy_full".to_owned()]);
        assert!(lookahead_ids(Some(&only_energy), &reached).is_empty());
        let with_cluster = BTreeSet::from([
            "operator_cluster".to_owned(),
            "trial_vector_energy".to_owned(),
        ]);
        assert_eq!(
            lookahead_ids(Some(&with_cluster), &reached),
            vec!["trial_vector_energy"]
        );
        let done = BTreeSet::from(["trial_vector_energy".to_owned()]);
        assert!(lookahead_ids(Some(&with_cluster), &done).is_empty());
    }

    #[test]
    fn lookahead_capture_matches_serial_capture_byte_for_byte() {
        use crate::ccm::convergence_capture::lookahead;
        let lanes = lookahead::FINITE.len() + lookahead::EXTENDED.len();
        for failing_trials in [false, true] {
            let (serial, taken) = lookahead_sequence(false, false, failing_trials);
            assert_eq!(taken, 0);
            let outcome = |id: &str| {
                serial
                    .iter()
                    .find(|(name, _)| name == &format!("outcome {id}"))
                    .unwrap()
                    .1
                    .clone()
            };
            for id in ["trial_vector_energy", "trial_vector_parity"] {
                let outcome = outcome(id);
                assert_eq!(
                    outcome.contains("trial coefficients do not match retained full matrix"),
                    failing_trials,
                    "{id}: {outcome}"
                );
            }
            // A failed diagnostic does not stop later ones.
            assert!(outcome("spectral_cluster_bound").contains("completed"));
            let compare = |label: &str, (outputs, taken): (Vec<(String, String)>, usize)| {
                assert_eq!(outputs.len(), serial.len(), "{label}");
                for (actual, expected) in outputs.iter().zip(&serial) {
                    assert_eq!(actual.0, expected.0, "{label}");
                    if actual.1 != expected.1 {
                        let at = actual
                            .1
                            .bytes()
                            .zip(expected.1.bytes())
                            .position(|(a, b)| a != b)
                            .unwrap_or(actual.1.len().min(expected.1.len()));
                        let window = |text: &str| {
                            text.get(at.saturating_sub(200)..(at + 200).min(text.len()))
                                .unwrap_or_default()
                                .to_owned()
                        };
                        panic!(
                            "{label}: {} differs at byte {at}:\n{}\n{}",
                            actual.0,
                            window(&actual.1),
                            window(&expected.1)
                        );
                    }
                }
                taken
            };
            // An application thread on the global pool, as in production.
            let taken = compare("global", lookahead_sequence(true, true, failing_trials));
            assert_eq!(taken, lanes);
            for threads in [1, 4] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                for wait in [true, false] {
                    let label = format!("{threads} threads, wait {wait}");
                    let taken = compare(
                        &label,
                        pool.install(|| lookahead_sequence(true, wait, failing_trials)),
                    );
                    if wait {
                        assert_eq!(taken, lanes, "{label}");
                    }
                }
            }
        }
    }
}
