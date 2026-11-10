//! Root-only recovery of an exact previously admitted CCM point source.
//!
//! This adapter authenticates retained payloads and their source identities.
//! It does not reload Tau, repeat its residual/ground proof, or upgrade the
//! original matrix assembly accuracy. The caller must preserve that original
//! journal and describe ground evidence as inherited rather than replayed.
use super::*;
use anyhow::ensure;

/// An immutable, authenticated retained even-sector source. Construction binds
/// its coefficients and state-resolution record to the original point payload.
pub struct RetainedSeededRootSource {
    params: CcmParams,
    cfg: HighPrecConfig,
    primary: PortableHighPrecResult,
    xi: Vec<Float>,
    length: Float,
    secular: ArtifactManifest,
    eigenstate_solver: CcmEigenstateSolver,
}

fn admit(
    manifest: &ArtifactManifest,
    payload: Option<&[u8]>,
    kind: &str,
    allowed: &[ContentDigest],
) -> Result<SemanticKeyEnvelope> {
    manifest.validate()?;
    ensure!(
        manifest.key.kind == kind
            && manifest.immutable
            && manifest.quality.admissible_rank() >= CacheQuality::Validated.admissible_rank()
            && allowed.contains(&manifest.content_digest),
        "retained root source is not an approved immutable validated {kind}"
    );
    if let Some(payload) = payload {
        ensure!(
            manifest.size_bytes == payload.len() as u64
                && ContentDigest::sha256(payload) == manifest.content_digest,
            "retained {kind} payload authentication failed"
        );
    }
    let semantic: SemanticKeyEnvelope = serde_json::from_str(
        manifest
            .tags
            .get("xc.semantic_key.v1")
            .context("missing retained semantic identity")?,
    )?;
    ensure!(
        semantic.artifact_kind == kind && semantic.digest()? == manifest.key.parameters_digest,
        "retained {kind} semantic identity mismatch"
    );
    // A canonical source, when present, must agree with its native envelope.
    xc_cache::retained_canonical_manifest(manifest)?;
    Ok(semantic)
}

impl RetainedSeededRootSource {
    /// Authenticate exact retained coefficients, the secular definition, and
    /// the original portable result. Tau is bound by immutable identity only;
    /// its original computation and state-selection proof are not rerun.
    #[allow(clippy::too_many_arguments)]
    pub fn from_payloads(
        params: &CcmParams,
        cfg: &HighPrecConfig,
        original: &PortableHighPrecResult,
        tau: &ArtifactManifest,
        eigenpair: &ArtifactManifest,
        eigenpair_payload: &[u8],
        secular: &ArtifactManifest,
        secular_payload: &[u8],
        allowed: &[ContentDigest],
    ) -> Result<Self> {
        ensure!(
            cfg.root_precision_policy == RootPrecisionPolicy::Adaptive,
            "retained root recovery requires explicit adaptive arithmetic"
        );
        cfg.validate_root_precision_policy()?;
        ensure!(
            cfg.effective_parity_policy() == CcmParityPolicy::EvenSector
                && cfg.ground_index_policy == CcmGroundIndexPolicy::Require,
            "retained root recovery currently requires the established even-sector source"
        );
        let tau_semantic = admit(tau, None, "ccm_tau_matrix", allowed)?;
        let ep_semantic = admit(
            eigenpair,
            Some(eigenpair_payload),
            "ccm_weil_eigenpair",
            allowed,
        )?;
        let sec_semantic = admit(
            secular,
            Some(secular_payload),
            "ccm_secular_source",
            allowed,
        )?;
        let cutoff = lambda_squared_cache_identity(params);
        for semantic in [&tau_semantic, &ep_semantic, &sec_semantic] {
            let p = &semantic.resolved_mathematical_parameters;
            ensure!(
                p["lambda_squared"] == cutoff
                    && p["n_modes"] == params.n_modes
                    && p["precision_bits"] == cfg.precision_bits,
                "retained source C/N/precision differs from root recovery request"
            );
        }
        let declared_tau: DependencyRef = serde_json::from_value(
            ep_semantic.resolved_mathematical_parameters["tau_source_identity"].clone(),
        )?;
        ensure!(
            declared_tau.key == tau.key
                && declared_tau.content_digest == tau.content_digest
                && declared_tau.required_quality == CacheQuality::Validated,
            "retained eigenpair belongs to a different Tau identity"
        );
        ensure!(
            xc_cache::manifest_depends_on(secular, eigenpair)?,
            "retained secular definition has a different eigenpair parent"
        );
        let ep: PortableWeilEigenpair = serde_json::from_slice(eigenpair_payload)?;
        let sec: PortableSecularSource = serde_json::from_slice(secular_payload)?;
        let solver = match ep.eigenstate_route.as_str() {
            "legacy_inverse_iteration" => CcmEigenstateSolver::LegacyInverseIteration,
            "shift_invert_krylov" => CcmEigenstateSolver::ShiftInvertKrylov,
            _ => bail!("unsupported retained eigenstate route"),
        };
        let mut resolved_cfg = cfg.clone();
        resolved_cfg.eigenstate_solver = solver;
        let (expected_ep, _) = weil_eigenpair_cache_identity(params, &resolved_cfg, tau)?;
        ensure!(
            expected_ep == ep_semantic,
            "retained eigenpair arithmetic policy differs"
        );
        ensure!(
            ep.schema_version
                == if solver == CcmEigenstateSolver::LegacyInverseIteration {
                    4
                } else {
                    5
                }
                && ep.lambda_squared == cutoff
                && ep.n_modes == params.n_modes
                && ep.precision_bits == cfg.precision_bits
                && payload_parity_matches(
                    ep.force_even,
                    ep.parity_policy,
                    CcmParityPolicy::EvenSector
                )
                && ep.ground_index.is_none(),
            "retained eigenpair shape or policy differs"
        );
        ensure!(
            sec.schema_version == 2
                && sec.lambda_squared == cutoff
                && sec.n_modes == params.n_modes
                && sec.precision_bits == cfg.precision_bits
                && payload_parity_matches(
                    sec.force_even,
                    sec.parity_policy,
                    CcmParityPolicy::EvenSector
                )
                && sec.eigenpair_content_digest == eigenpair.content_digest.0
                && sec.eigenpair_semantic_digest == eigenpair.key.parameters_digest.0
                && sec.normalization == "sum_xi_equals_sqrt_log_lambda_squared",
            "retained secular payload does not describe this exact eigenpair"
        );
        let sec_params = &sec_semantic.resolved_mathematical_parameters;
        ensure!(
            sec_params["eigenpair_content_digest"] == eigenpair.content_digest.0
                && sec_params["eigenpair_semantic_digest"] == eigenpair.key.parameters_digest.0,
            "retained secular semantic parents differ"
        );
        let expected_sec = SemanticKeyEnvelope {
            schema_version: 1,
            artifact_kind: "ccm_secular_source".into(),
            mathematical_semantics_version: "ccm-secular-source-v0.13.0-v2".into(),
            resolved_mathematical_parameters: serde_json::json!({
                "lambda_squared": cutoff, "n_modes":params.n_modes,
                "precision_bits":cfg.precision_bits,"force_even":true,
                "eigenpair_content_digest":eigenpair.content_digest.0,
                "eigenpair_semantic_digest":eigenpair.key.parameters_digest.0,
            }),
            normalization: Some("sum_xi_equals_sqrt_log_lambda_squared".into()),
            target: Some("ccm_secular_function".into()),
            subspace: CcmParityPolicy::EvenSector.semantic_subspace(),
            source_data_identities: BTreeMap::new(),
            algorithm_semantics: Some("xi_hat_exponential_sum".into()),
        };
        ensure!(
            sec_semantic == expected_sec,
            "retained secular arithmetic/normalization contract differs"
        );
        let runtime = original.to_runtime()?;
        let xi = parse_hp_vector(&ep.eigenvector, cfg.precision_bits)?;
        let length = log_lambda_sq_hp(params, cfg.precision_bits)?;
        validate_eigenstate_contract(
            &xi,
            &length,
            cfg.precision_bits,
            CcmParityPolicy::EvenSector,
        )?;
        let xi_portable = xi
            .iter()
            .map(xc_numerics::fmt::PortableHpFloat::from_float)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        ensure!(
            original.precision_bits == cfg.precision_bits
                && xi.len() == params.matrix_size()
                && xi_portable == original.xi
                && runtime.weil_min_eigenvalue
                    == parse_hp_scalar(&ep.eigenvalue, cfg.precision_bits)?
                && original.inverse_iteration_diagnostics == ep.inverse_iteration
                && original.stored_state_resolution == ep.stored_state_resolution
                && original.ground_index_admission.is_none(),
            "original portable state or inherited ground evidence differs from retained eigenpair"
        );
        let resolution = ep
            .stored_state_resolution
            .as_ref()
            .context("retained source lacks stored resolution")?;
        ensure!(
            resolution.arithmetic == stored_resolution::ARITHMETIC
                && resolution.separation_unresolved_reason.is_none(),
            "retained ground resolution is unresolved or has unsupported arithmetic"
        );
        let gap = parse_hp_scalar(
            resolution
                .selected_gap_lower
                .as_deref()
                .context("missing finite gap")?,
            cfg.precision_bits,
        )?;
        let angle = parse_hp_scalar(
            resolution
                .residual_angle_upper
                .as_deref()
                .context("missing finite residual angle")?,
            cfg.precision_bits,
        )?;
        ensure!(
            gap.is_finite() && angle.is_finite() && gap > 0 && angle >= 0,
            "retained ground resolution is invalid"
        );
        Ok(Self {
            params: params.clone(),
            cfg: resolved_cfg,
            primary: original.clone(),
            xi,
            length,
            secular: secular.clone(),
            eigenstate_solver: solver,
        })
    }

    /// Produce a new versioned adaptive root artifact from the unchanged exact
    /// source. The ordinary production decoder replays every retained root's
    /// directed local witness and correction gate. Nonconvergence is retained,
    /// never promoted; callers must require every requested outcome to converge.
    pub fn refine(
        &self,
        first: usize,
        seeds: &[Float],
        dataset: &ReferenceZeroDatasetIdentity,
        cache: &ArtifactCacheContext<'_>,
    ) -> Result<(Vec<EigenvalueResult>, ArtifactManifest)> {
        ensure!(
            cache.requested_assurance == xc_core::AssuranceLevel::Computed,
            "retained root recovery does not independently recertify original matrix/state"
        );
        validate_reference_seed_dataset(
            RootArtifactMode::ReferenceSeededRefinement,
            Some(dataset),
            first,
            seeds.len(),
        )?;
        ensure!(
            seeds.iter().all(|x| x.prec() == self.cfg.precision_bits),
            "reference seed precision differs from retained source"
        );
        xc_numerics::hp_runtime::run_hp(|| {
            let (roots, manifest, _) = resolve_root_range_via_cache(
                &self.params,
                &self.cfg,
                self.eigenstate_solver,
                &self.length,
                &self.xi,
                first,
                seeds,
                &self.secular,
                cache,
                RootArtifactMode::ReferenceSeededRefinement,
                Some(dataset),
                RootWindowSemantics::strict_positive(seeds.len()),
            )?;
            Ok((roots, manifest))
        })
    }

    /// Original portable state for explicit source-equality checks. Its old
    /// root outcomes retain their original failure statuses.
    pub fn original(&self) -> &PortableHighPrecResult {
        &self.primary
    }
}
