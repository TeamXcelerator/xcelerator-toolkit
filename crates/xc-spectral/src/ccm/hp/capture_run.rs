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
    Ok(CapturedDiagnostic::new(
        &values,
        records.into_iter().map(|r| r.manifest).collect(),
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
                assembly_policy: "ccm-weil-form-v0.13.0-v2; corrected symmetric Tau".into(),
                quadrature_policy: format!(
                    "frequency-aware HP GL; configured quad_points={}",
                    cfg.quad_points
                ),
            };
            if let Err(e) = crate::ccm::research_cohort::register(&registration) {
                eprintln!("research cohort registration unavailable: {e}");
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
    /// Outcome-aware adapter. Missing external data remains Missing in a receipt;
    /// numerical nonacceptance stays in the retained measurement.
    pub fn capture_diagnostic_outcome(
        &mut self,
        id: &str,
        options: &CcmResearchCaptureOptions,
        cache: &ArtifactCacheContext<'_>,
    ) -> std::result::Result<CapturedDiagnostic, xc_cache::CaptureFailure> {
        let result = self
            .capture_diagnostic(id, options, cache)
            .map_err(xc_cache::CaptureFailure::failed)?;
        if result.value["data"]["outcome"] == "missing_input" {
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
                    input.precision_bits = input.precision_bits.max(prepared.precision_bits);
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
                let v = crate::ccm::extended_research::source_unit(&state, state.precision);
                let n = v.len();
                let action = matrix
                    .par_chunks(n)
                    .map(|row| {
                        lossless_hp_decimal(
                            &(-crate::ccm::retained_evidence::dot(row, &v, state.precision)),
                        )
                    })
                    .collect();
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
        let n = state.coefficients.len();
        let l = log_lambda_sq_hp(&self.params, p);
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
        let v = crate::ccm::extended_research::source_unit(state, p);
        let pi2 = pi(p).square() * 16u32;
        let l2 = l.clone().square();
        let pref = (Float::with_val(p, &l) / 4u32).sinh().square() * 32u32 * &l;
        let rows = (0..n)
            .into_par_iter()
            .map(|r| {
                let ri = r as i64 - state.modes as i64;
                let mut pole = Float::with_val(p, 0);
                let mut arch = Float::with_val(p, 0);
                let mut total = Float::with_val(p, 0);
                for (c, vc) in v.iter().enumerate() {
                    let ci = c as i64 - state.modes as i64;
                    let numerator = Float::with_val(p, &l2) - Float::with_val(p, &pi2) * ri * ci;
                    let denominator = (Float::with_val(p, &pi2) * ci * ci + &l2)
                        * (Float::with_val(p, &pi2) * ri * ri + &l2);
                    pole += Float::with_val(p, &pref) * numerator / denominator * vc;
                    let av = if ri == ci {
                        (Float::with_val(p, &integrals.gamma[ri.unsigned_abs() as usize])
                            - &integrals.beta[ri.unsigned_abs() as usize])
                            * 2u32
                    } else {
                        (signed_alpha(&integrals.alpha, ci, p)
                            - signed_alpha(&integrals.alpha, ri, p))
                            / (ri - ci)
                    };
                    arch -= av * vc;
                    total += Float::with_val(p, &self.source.tau[r * n + c]) * vc;
                }
                let prime = Float::with_val(p, &total) - &pole - &arch;
                (pole, arch, prime)
            })
            .collect::<Vec<_>>();
        let notes="signed action on center-oriented unit coefficient state; pole and archimedean from exact retained primitive ancestry; prime=Tau-pole-archimedean is an algebraic reconstruction, not an independent prime-closure validation";
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
        force_symmetric(&mut tau, self.params.matrix_size());
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
        if let Some(error) = &self.sector_error {
            bail!("sector diagnostic unavailable: {error}");
        }
        if self.sectors.is_some() {
            return Ok(());
        }
        let options = options
            .sector_analysis
            .ok_or_else(|| anyhow::anyhow!("sector analysis not requested"))?;
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

    /// Attempt one primary diagnostic. Callers convert errors to explicit
    /// receipt outcomes and continue other independent requests. Measurements
    /// include exact authenticated artifact manifests, on fresh and warm runs.
    pub fn capture_diagnostic(
        &mut self,
        id: &str,
        options: &CcmResearchCaptureOptions,
        cache: &ArtifactCacheContext<'_>,
    ) -> Result<CapturedDiagnostic> {
        xc_numerics::hp_runtime::run_hp(|| self.capture_inner(id, options, cache))
    }
    fn capture_inner(
        &mut self,
        id: &str,
        options: &CcmResearchCaptureOptions,
        cache: &ArtifactCacheContext<'_>,
    ) -> Result<CapturedDiagnostic> {
        if id == "u_flow_response" {
            if let Some(saved) = &self.uflow_capture {
                return Ok(CapturedDiagnostic::new(
                    &saved.value,
                    saved.sources.clone(),
                )?);
            }
        }
        let complete = id.ends_with("_full")
            || crate::ccm::convergence_capture::DIAGNOSTICS.contains(&id)
            || crate::ccm::research_completion::DIAGNOSTICS.contains(&id);
        let id = id.strip_suffix("_full").unwrap_or(id);
        if crate::ccm::extended_research::DIAGNOSTICS.contains(&id)
            || crate::ccm::convergence_capture::DIAGNOSTICS.contains(&id)
            || crate::ccm::research_completion::DIAGNOSTICS.contains(&id)
        {
            use crate::ccm::extended_research::*;
            use crate::ccm::retained_evidence::{RetainedMatrix, RetainedRoots};
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
                            self.prepared_reference = Some(crate::ccm::research_completion::ReferencePreparation {
                                schema_version: 1,
                                finite_reference: Some(reference.clone()),
                                sampled_reference: input.target.clone(),
                                lambda_squared: input.lambda_squared.clone(),
                                precision_bits: input.precision_bits,
                                definition_digest: input.definition_digest.clone(),
                                approximation_scope: format!("{}; signed jets refer to this explicitly finite Fourier projection, extended by zero outside the run window", input.approximation_scope),
                                weighted_atoms: vec![], atom_coordinate: None, atom_coverage: None,
                                tail_form: None, tail_recipe: None, completion: None,
                            });
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
                                self.extended_input_error =
                                    Some(format!("reference preparation: {e}"))
                            }
                        }
                    }
                }
            }
            if complete {
                self.prepare_run_once_inputs(options, cache)?;
            }
            // Source-only groups remain available even when optional input loading failed.
            if !matches!(
                id,
                "compactness"
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
                "arithmetic_energy"
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
                "directional_response"
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
                        Ok(prepared) => {
                            if let Some(existing) = &mut self.extended_inputs {
                                if existing.reference_jets.is_empty() {
                                    existing.reference_jets = prepared.reference_jets;
                                }
                                existing.precision_bits =
                                    existing.precision_bits.max(prepared.precision_bits);
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
            let mut options = self
                .extended_options
                .clone()
                .unwrap_or_else(|| ExtensionOptions::for_source(&state));
            if self.extended_options.is_none() {
                options.working_precision_bits = options
                    .working_precision_bits
                    .max(
                        self.extended_inputs
                            .as_ref()
                            .map_or(0, |i| i.precision_bits.saturating_add(64)),
                    )
                    .max(
                        roots
                            .as_ref()
                            .map_or(0, |r| r.dataset.precision_bits.saturating_add(64)),
                    );
            }
            if complete {
                if self.extended_options.is_none() {
                    let policy =
                        crate::ccm::capture_runtime::CaptureResourcePolicy::from_environment()?;
                    options.maximum_estimated_output_bytes = policy.maximum_output_bytes;
                    options.maximum_working_bytes = Some(policy.maximum_working_bytes);
                }
                if matches!(id, "directional_response" | "root_transport") {
                    options.maximum_directional_rows =
                        roots.as_ref().map_or(0, |r| r.dataset.points.len());
                }
            }
            let input = if id == "compactness" {
                None
            } else {
                self.extended_inputs.as_ref()
            };
            return Ok(CapturedDiagnostic::from_cached(capture_extended(
                id,
                &state,
                matrix.as_ref(),
                roots.as_ref(),
                input,
                &options,
                &self.run_once_sources,
                cache,
            )?)?);
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
        if matches!(
            id,
            "evenness" | "sector_analysis" | "sector_gap_certificate"
        ) {
            self.sectors(options, cache)?;
            let sectors = self.sectors.as_ref().expect("resolved sectors");
            let record = self.sector_record.as_ref().expect("recorded sectors");
            if id == "evenness" {
                let value =
                    evenness_from_sector_gap(&self.params, self.cfg.precision_bits, &sectors.gap)?;
                return Ok(CapturedDiagnostic::new(
                    &serde_json::json!({"method":"winning_parity_sector_lift", "evenness_deviation":value.evenness_deviation.to_string(), "natural_eigenvalue":value.natural_eigenvalue.to_string(), "forced_eigenvalue":value.forced_eigenvalue.to_string()}),
                    record.sources.clone(),
                )?);
            }
            if id == "sector_gap_certificate" {
                let certification = options
                    .sector_gap_certification
                    .ok_or_else(|| anyhow::anyhow!("sector certification not requested"))?;
                let certificate = certify_sector_gap_from_resolution(
                    &self.params,
                    &self.cfg,
                    certification,
                    sectors,
                    Some(cache),
                )?;
                return Ok(CapturedDiagnostic::new(
                    &certificate,
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
            "root_certificate" => "ccm_root_certificate",
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
        if matches!(id, "prime_power_response" | "u_flow_response") {
            if cfg.effective_parity_policy() != CcmParityPolicy::EvenSector {
                bail!("response requires an isolated even-sector state; primary parity preserved");
            }
            let l = log_lambda_sq_hp(p, cfg.precision_bits);
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
                &log_lambda_sq_hp(p, cfg.precision_bits),
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
            let certificate = certify_roots_from_retained_source(
                p,
                cfg,
                &primary.xi,
                source.secular_manifest.as_ref(),
                certification,
                Some(&observed),
            )?;
            reconcile_computed_roots_with_certificate(primary, &certificate)?;
            return Ok(CapturedDiagnostic::new(
                &certificate,
                vec![required(&source.secular_manifest)?],
            )?);
        } else if id == "distance_profile" {
            let distance = options
                .distance_capture
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("profile capture not requested"))?;
            let rule = distance
                .rules
                .first()
                .ok_or_else(|| anyhow::anyhow!("profile capture requires a grid convention"))?;
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
            crate::distance::hp::capture_ccm_distance_with_derived_via_cache(
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
        }
        let result = recorded(sink.finish()?)?;
        if id == "u_flow_response" {
            self.uflow_capture = Some(CapturedDiagnostic::new(
                &result.value,
                result.sources.clone(),
            )?);
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

    fn retained_run_round_trip(with_staging: bool, wider_window: bool) {
        use xc_cache::{
            ArtifactExecutionCacheMode, CacheLayer, CachePolicy, CacheResolver, CacheVisibility,
            ZipJsonFilesystemCacheStore as FilesystemCacheStore,
        };
        let root = std::env::temp_dir().join(format!(
            "ccm-retained-run-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
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
}
