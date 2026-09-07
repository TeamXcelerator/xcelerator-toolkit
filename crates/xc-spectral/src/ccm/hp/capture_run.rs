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
        let sink = RecordingSink::new(cache, &["ccm_weil_eigenpair"]);
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
        let eigenpair = sink.finish()?.into_iter().find(|r| {
            source.eigenpair_manifest.as_ref().is_some_and(|m| {
                m.key == r.manifest.key && m.content_digest == r.manifest.content_digest
            })
        });
        Ok(Self {
            params: params.clone(),
            cfg: cfg.clone(),
            primary,
            source,
            eigenpair,
            sectors: None,
            sector_record: None,
            sector_error: None,
        })
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
        recorded(sink.finish()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn retained_run_survives_diagnostic_failure_and_authenticates_warm_sources() {
        retained_run_round_trip(false);
    }

    #[test]
    fn retained_run_captures_cold_artifacts_with_encoded_publication_staging() {
        retained_run_round_trip(true);
    }

    fn retained_run_round_trip(with_staging: bool) {
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
            current_toolkit_version: ToolkitVersion::parse("0.15.0").unwrap(),
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
        let initial = PortableHighPrecResult::from_runtime(run.primary()).unwrap();
        let mut options = super::super::super::capture::CcmCapturePlan::ultra(2, 17)
            .unwrap()
            .primary_options()
            .unwrap();
        options.distance_capture = Some(CcmDistanceCaptureOptions::default_convention(32, 32));
        if with_staging {
            // Fresh state/matrix notifications must survive encoded production too.
            assert_eq!(run.retained_even_sources(&cache).unwrap().1.len(), 1);
            for id in [
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
        std::fs::remove_dir_all(root).unwrap();
    }
}
