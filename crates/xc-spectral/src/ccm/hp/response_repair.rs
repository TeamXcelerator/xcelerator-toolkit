//! Offline repair of the normalization cancellation in schema-2 responses.
//! Only root-velocity fields change. No matrix assembly, eigenstate solve, or
//! bordered tangent solve is performed, and source assurance is not upgraded.
use super::*;
use xc_cache::{LocalShardJson, REMOTE_CANONICAL_MANIFEST_TAG};

/// Corrected bytes and their ordinary production semantic key. The caller must
/// retain the old artifact and publish the result additively, at the same
/// visibility. This does not certify the retained numerical sources.
pub struct RepairedResponse {
    pub payload: Vec<u8>,
    pub semantic_key: SemanticKeyEnvelope,
}

fn authenticate(source: &LocalShardJson) -> Result<xc_cache::CanonicalArtifactManifest> {
    source.manifest.validate()?;
    if ContentDigest::sha256(&source.payload) != source.manifest.content_digest
        || source.manifest.size_bytes != source.payload.len() as u64
        || source.manifest.quality.admissible_rank() < CacheQuality::Validated.admissible_rank()
    {
        bail!("retained repair source has invalid bytes, size, or quality");
    }
    let canonical: xc_cache::CanonicalArtifactManifest = serde_json::from_str(
        source
            .manifest
            .tags
            .get(REMOTE_CANONICAL_MANIFEST_TAG)
            .ok_or_else(|| {
                anyhow::anyhow!("repair requires an authenticated canonical shard source")
            })?,
    )?;
    canonical.validate()?;
    if canonical.digest()? != source.canonical_manifest_digest
        || canonical.semantic_digest != source.manifest.key.parameters_digest
        || canonical.semantic_key.artifact_kind != source.manifest.key.kind
        || canonical.canonical_payload.ordered_items.len() != 1
        || canonical.canonical_payload.ordered_items[0].content_digest
            != source.manifest.content_digest
        || canonical.canonical_payload.ordered_items[0].size_bytes != source.payload.len() as u64
    {
        bail!("retained repair canonical source binding mismatch");
    }
    Ok(canonical)
}

fn finite_vector(values: &[String], dimension: usize, bits: u32) -> Result<Vec<Float>> {
    if values.len() != dimension {
        bail!("retained repair vector dimension mismatch");
    }
    let result = parse_hp_vector(values, bits)?;
    if result.iter().any(|v| !v.is_finite()) {
        bail!("nonfinite retained repair vector");
    }
    Ok(result)
}

fn root_values(roots: &[PortableCcmResponseRoot], bits: u32) -> Result<Vec<Option<Float>>> {
    roots
        .iter()
        .enumerate()
        .map(|(index, root)| {
            if root.window_position != index + 1
                || !matches!(
                    root.status.as_str(),
                    "converged" | "stagnated" | "approximate" | "failed"
                )
                || (root.status == "failed") != root.value.is_none()
            {
                bail!("invalid retained root status or position");
            }
            root.value
                .as_ref()
                .map(|v| {
                    let value = parse_hp_scalar(v, bits)?;
                    if !value.is_finite() {
                        bail!("nonfinite retained root");
                    }
                    Ok(value)
                })
                .transpose()
        })
        .collect()
}

fn fixed_responses(
    unit: &[Float],
    tangent: &[Float],
    poles: &[Float],
    roots: &[Option<Float>],
    bits: u32,
) -> Result<Vec<Option<String>>> {
    roots
        .iter()
        .map(|root| {
            root.as_ref()
                .map(|root| {
                    let result =
                        prime_power_root_velocity_response(unit, tangent, poles, root, bits)?;
                    if !result.is_finite() {
                        bail!("nonfinite repaired root response");
                    }
                    Ok(lossless_hp_decimal(&result))
                })
                .transpose()
        })
        .collect()
}

fn moving_responses(
    unit: &[Float],
    tangent: &[Float],
    poles: &[Float],
    velocities: &[Float],
    roots: &[Option<Float>],
    bits: u32,
) -> Result<Vec<Option<String>>> {
    roots
        .iter()
        .map(|root| {
            root.as_ref()
                .map(|root| {
                    let result = secular_root_velocity_response(
                        unit, tangent, poles, velocities, root, bits,
                    )?;
                    if !result.is_finite() {
                        bail!("nonfinite repaired moving-pole response");
                    }
                    Ok(lossless_hp_decimal(&result))
                })
                .transpose()
        })
        .collect()
}

/// Repair an old schema-2 response using its exact original eigenpair and its
/// retained L2 tangent vectors, at the original MPFR working precision.
/// Authentication includes raw payload hashes, canonical identities, the exact
/// eigenpair dependency, and the shared numerical configuration. Already-v3
/// sources are accepted only if replay reproduces their bytes exactly.
pub fn repair_retained_response(
    response: &LocalShardJson,
    eigenpair: &LocalShardJson,
) -> Result<RepairedResponse> {
    let original = authenticate(response)?;
    let state_manifest = authenticate(eigenpair)?;
    if state_manifest.semantic_key.artifact_kind != "ccm_weil_eigenpair"
        || !original.canonical_payload.dependencies.iter().any(|d| {
            d.semantic_digest == state_manifest.semantic_digest
                && d.manifest_digest == eigenpair.canonical_manifest_digest
                && d.payload_digest == state_manifest.payload_digest
        })
    {
        bail!("retained response does not name this exact eigenpair dependency");
    }
    let state: PortableWeilEigenpair = serde_json::from_slice(&eigenpair.payload)?;
    let mut semantic = original.semantic_key;
    let (old_version, new_version) = match semantic.artifact_kind.as_str() {
        "ccm_prime_power_response_analysis" => (
            "ccm-prime-power-response-v0.14.1-v2",
            "ccm-prime-power-response-v0.15.0-v3",
        ),
        "ccm_u_flow_response_analysis" => (
            "ccm-u-flow-response-v0.14.1-v2",
            "ccm-u-flow-response-v0.15.0-v3",
        ),
        _ => bail!("unsupported retained response kind"),
    };
    if semantic.mathematical_semantics_version != old_version
        && semantic.mathematical_semantics_version != new_version
    {
        bail!("unsupported response semantics; schema-1 responses do not retain repair inputs");
    }
    let was_current = semantic.mathematical_semantics_version == new_version;
    let metadata: serde_json::Value = serde_json::from_slice(&response.payload)?;
    let bits = state.precision_bits;
    let dimension = state
        .n_modes
        .checked_mul(2)
        .and_then(|n| n.checked_add(1))
        .ok_or_else(|| anyhow::anyhow!("repair dimension overflow"))?;
    if !(64..=1_000_000).contains(&bits)
        || state.n_modes > i64::MAX as usize
        || metadata["schema_version"] != 2
        || metadata["lambda_squared"] != state.lambda_squared
        || metadata["n_modes"] != state.n_modes
        || metadata["dimension"] != dimension
        || metadata["precision_bits"] != bits
        || metadata["force_even"] != state.force_even
        || metadata.get("parity_policy") != serde_json::to_value(&state)?.get("parity_policy")
        || metadata["eigenpair_content_digest"] != eigenpair.manifest.content_digest.0
    {
        bail!("retained response/eigenpair configuration mismatch");
    }
    let response_eigenvalue = parse_hp_scalar(
        metadata["state_eigenvalue"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing retained state eigenvalue"))?,
        bits,
    )?;
    let state_eigenvalue = parse_hp_scalar(&state.eigenvalue, bits)?;
    if !state_eigenvalue.is_finite() || response_eigenvalue != state_eigenvalue {
        bail!("retained response/eigenpair eigenvalue mismatch");
    }
    drop(metadata);
    let xi = finite_vector(&state.eigenvector, dimension, bits)?;
    let norm = deterministic_l2_norm_hp(&xi, bits);
    if !norm.is_finite() || norm.is_zero() {
        bail!("invalid retained eigenstate norm");
    }
    let unit = xi
        .iter()
        .map(|x| Float::with_val(bits, x) / &norm)
        .collect::<Vec<_>>();
    let params = match state.lambda_squared.parse::<u64>() {
        Ok(c) => CcmParams::from_lambda_sq_integer(c, state.n_modes),
        Err(_) => {
            CcmParams::from_lambda_sq_fractional(state.lambda_squared.parse()?, state.n_modes)
        }
    };
    if lambda_squared_cache_identity(&params) != state.lambda_squared {
        bail!("unsupported retained cutoff identity");
    }
    let l = log_lambda_sq_hp(&params, bits);
    if !l.is_finite() || l <= 0 {
        bail!("invalid retained cutoff");
    }
    let (poles, pole_velocities) = ccm_secular_poles_and_u_velocities(&l, state.n_modes, bits);
    let payload = if semantic.artifact_kind == "ccm_prime_power_response_analysis" {
        let mut data: PortablePrimePowerResponseAnalysis =
            serde_json::from_slice(&response.payload)?;
        let roots = root_values(&data.roots, bits)?;
        for event in &mut data.events {
            if event.root_velocity_responses.len() != roots.len() {
                bail!("invalid retained response shape");
            }
            let tangent = finite_vector(&event.l2_eigenvector_velocity_response, dimension, bits)?;
            if lossless_hp_decimal(&deterministic_l2_norm_hp(&tangent, bits))
                != event.l2_eigenvector_velocity_response_norm
            {
                bail!("retained tangent norm mismatch");
            }
            event.root_velocity_responses = fixed_responses(&unit, &tangent, &poles, &roots, bits)?;
        }
        serde_json::to_vec(&data)?
    } else {
        let mut data: PortableUFlowResponseAnalysis = serde_json::from_slice(&response.payload)?;
        let roots = root_values(&data.roots, bits)?;
        if data.secular_pole_motion_root_velocity_responses.len() != roots.len()
            || data.total_moving_pole_root_velocity_responses.len() != roots.len()
        {
            bail!("invalid retained moving-pole response shape");
        }
        let mut total = None;
        for channel in &mut data.channels {
            if channel.fixed_pole_root_velocity_responses.len() != roots.len() {
                bail!("invalid retained channel shape");
            }
            let tangent =
                finite_vector(&channel.l2_eigenvector_velocity_response, dimension, bits)?;
            if lossless_hp_decimal(&deterministic_l2_norm_hp(&tangent, bits))
                != channel.l2_eigenvector_velocity_response_norm
            {
                bail!("retained tangent norm mismatch");
            }
            channel.fixed_pole_root_velocity_responses =
                fixed_responses(&unit, &tangent, &poles, &roots, bits)?;
            if channel.channel == "tau_total" {
                if total.is_some() {
                    bail!("duplicate total tangent");
                }
                total = Some(tangent);
            }
        }
        let total = total.ok_or_else(|| anyhow::anyhow!("missing total tangent"))?;
        let zero = vec![Float::with_val(bits, 0); dimension];
        data.secular_pole_motion_root_velocity_responses =
            moving_responses(&unit, &zero, &poles, &pole_velocities, &roots, bits)?;
        data.total_moving_pole_root_velocity_responses =
            moving_responses(&unit, &total, &poles, &pole_velocities, &roots, bits)?;
        serde_json::to_vec(&data)?
    };
    if was_current && response.payload != payload {
        bail!("current response failed exact retained replay");
    }
    semantic.mathematical_semantics_version = new_version.into();
    Ok(RepairedResponse {
        payload,
        semantic_key: semantic,
    })
}

#[cfg(test)]
fn fixture_source(
    kind: &str,
    version: &str,
    bytes: Vec<u8>,
    dependencies: Vec<xc_cache::PayloadDependencyIdentity>,
) -> LocalShardJson {
    use xc_cache::*;
    let semantic = SemanticKeyEnvelope {
        schema_version: 1,
        artifact_kind: kind.into(),
        mathematical_semantics_version: version.into(),
        resolved_mathematical_parameters: serde_json::json!({}),
        normalization: None,
        target: None,
        subspace: None,
        source_data_identities: BTreeMap::new(),
        algorithm_semantics: None,
    };
    let digest = ContentDigest::sha256(&bytes);
    let payload = CanonicalPayloadEnvelope {
        schema_version: 1,
        scalar_backend: "canonical_json".into(),
        precision_bits: Some(192),
        scalar_representation: "canonical-json-utf8-v1".into(),
        dimensions: vec![],
        endianness: "not-applicable".into(),
        special_value_encoding: "decimal-string-or-json-number-v1".into(),
        ordered_items: vec![LogicalPayloadItem {
            normalized_path: "payload.json".into(),
            content_digest: digest.clone(),
            size_bytes: bytes.len() as u64,
        }],
        dependencies,
    };
    let transport = ContentDigest::sha256(b"fixture transport");
    let canonical = CanonicalArtifactManifest {
        schema_version: 1,
        artifact_family: if kind == "ccm_weil_eigenpair" {
            "weil-states"
        } else {
            "ccm-evidence"
        }
        .into(),
        semantic_digest: semantic.digest().unwrap(),
        semantic_key: semantic.clone(),
        payload_digest: payload.digest().unwrap(),
        canonical_payload: payload,
        transport_digests: vec![transport.clone()],
        resolved_mathematical_configuration_digest: ContentDigest(
            xc_core::research_digest(&serde_json::json!({})).unwrap().0,
        ),
        producer_toolkit_version: ToolkitVersion::parse("0.15.0").unwrap(),
        minimum_reader_version: ToolkitVersion::parse("0.14.1").unwrap(),
        maximum_reader_version: None,
        requested_assurance: xc_core::AssuranceLevel::Computed,
        claim_scope: "test fixture".into(),
        assumptions: vec![],
    };
    let manifest = ArtifactManifest {
        schema_version: 1,
        key: ArtifactKey {
            kind: kind.into(),
            logical_key: format!("test/{kind}"),
            parameters_digest: semantic.digest().unwrap(),
        },
        content_digest: digest.clone(),
        size_bytes: bytes.len() as u64,
        objects: vec![CacheObjectRef {
            content_digest: digest,
            size_bytes: bytes.len() as u64,
        }],
        created_unix_seconds: 0,
        producer_toolkit_version: canonical.producer_toolkit_version.clone(),
        minimum_reader_version: canonical.minimum_reader_version.clone(),
        maximum_reader_version: None,
        quality: CacheQuality::CrossChecked,
        visibility: CacheVisibility::Private,
        immutable: true,
        dependencies: vec![],
        tags: BTreeMap::from([(
            REMOTE_CANONICAL_MANIFEST_TAG.into(),
            serde_json::to_string(&canonical).unwrap(),
        )]),
        provenance_digest: Some(canonical.digest().unwrap()),
    };
    LocalShardJson {
        manifest,
        payload: bytes,
        canonical_manifest_digest: canonical.digest().unwrap(),
        transport_digest: transport,
    }
}

/// Exercise the repair against freshly computed and warm-validated fixtures.
#[cfg(test)]
pub(super) fn check_fresh_response_repair(
    mut prime: PortablePrimePowerResponseAnalysis,
    mut flow: PortableUFlowResponseAnalysis,
    xi: &[Float],
    eigenvalue: &Float,
) {
    let state = PortableWeilEigenpair {
        schema_version: 3,
        lambda_squared: prime.lambda_squared.clone(),
        n_modes: prime.n_modes,
        precision_bits: prime.precision_bits,
        force_even: prime.force_even,
        parity_policy: prime.parity_policy,
        eigenstate_route: legacy_eigenstate_route_name(),
        eigenvalue: eigenvalue.to_string(),
        eigenvector: xi.iter().map(Float::to_string).collect(),
        inverse_iteration: PortableInverseIterationDiagnostics {
            configured_step_limit: 1,
            unshifted_steps: 1,
            unshifted_converged: true,
            final_relative_rayleigh_change: Some("0".into()),
            shifted_refinement: "not_attempted".into(),
            final_relative_residual_norm: "0".into(),
        },
        shift_invert_krylov: None,
    };
    let state = fixture_source(
        "ccm_weil_eigenpair",
        "test-state",
        serde_json::to_vec(&state).unwrap(),
        vec![],
    );
    prime.eigenpair_content_digest = state.manifest.content_digest.0.clone();
    flow.eigenpair_content_digest = state.manifest.content_digest.0.clone();
    let canonical = authenticate(&state).unwrap();
    let dep = xc_cache::PayloadDependencyIdentity {
        artifact_family: "weil-states".into(),
        semantic_digest: canonical.semantic_digest,
        manifest_digest: state.canonical_manifest_digest.clone(),
        payload_digest: canonical.payload_digest,
    };
    let fresh_prime = serde_json::to_vec(&prime).unwrap();
    let fresh_flow = serde_json::to_vec(&flow).unwrap();
    for event in &mut prime.events {
        for x in event.root_velocity_responses.iter_mut().flatten() {
            *x = "123".into();
        }
    }
    for channel in &mut flow.channels {
        for x in channel
            .fixed_pole_root_velocity_responses
            .iter_mut()
            .flatten()
        {
            *x = "123".into();
        }
    }
    for x in flow
        .secular_pole_motion_root_velocity_responses
        .iter_mut()
        .chain(&mut flow.total_moving_pole_root_velocity_responses)
        .flatten()
    {
        *x = "123".into();
    }
    for (kind, version, bytes, expected) in [
        (
            "ccm_prime_power_response_analysis",
            "ccm-prime-power-response-v0.14.1-v2",
            serde_json::to_vec(&prime).unwrap(),
            fresh_prime,
        ),
        (
            "ccm_u_flow_response_analysis",
            "ccm-u-flow-response-v0.14.1-v2",
            serde_json::to_vec(&flow).unwrap(),
            fresh_flow,
        ),
    ] {
        let mut old = fixture_source(kind, version, bytes.clone(), vec![dep.clone()]);
        let corrected = repair_retained_response(&old, &state).unwrap();
        assert_eq!(
            corrected.payload, expected,
            "repair must reproduce fresh producer bytes"
        );
        assert_eq!(
            old.payload, bytes,
            "original retained bytes must be preserved"
        );
        let current = fixture_source(
            kind,
            &corrected.semantic_key.mathematical_semantics_version,
            corrected.payload,
            vec![dep.clone()],
        );
        assert_eq!(
            repair_retained_response(&current, &state).unwrap().payload,
            expected
        );
        old.payload.push(b' ');
        assert!(
            repair_retained_response(&old, &state).is_err(),
            "tampered bytes must fail"
        );
        let unbound = fixture_source(kind, version, bytes.clone(), vec![]);
        assert!(
            repair_retained_response(&unbound, &state).is_err(),
            "unbound source must fail"
        );
        let mut bad: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        bad["n_modes"] = serde_json::json!(2);
        let invalid = fixture_source(
            kind,
            version,
            serde_json::to_vec(&bad).unwrap(),
            vec![dep.clone()],
        );
        assert!(
            repair_retained_response(&invalid, &state).is_err(),
            "shape mismatch must fail"
        );
    }
}
