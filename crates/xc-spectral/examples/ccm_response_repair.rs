//! Retained response repair. Offline preparation and explicit additive publication
//! are separate commands. See docs/CCM_RESPONSE_REPAIR.md.
#[cfg(feature = "hp")]
mod enabled {
    use anyhow::{bail, Context, Result};
    use serde::{Deserialize, Serialize};
    use std::{
        fs,
        io::Write,
        path::{Path, PathBuf},
    };
    use xc_cache::*;
    use xc_core::{CancellationToken, ResourcePolicy};
    use xc_spectral::ccm::hp::response_repair::repair_retained_response;

    #[derive(Deserialize)]
    #[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
    enum Request {
        Response {
            response: PathBuf,
            eigenpair: PathBuf,
            output: PathBuf,
            read: LocalShardReadOptions,
        },
        Receipt {
            receipt: PathBuf,
            repairs: Vec<Replacement>,
            output: PathBuf,
            read: LocalShardReadOptions,
        },
        Publish {
            drafts: Vec<PathBuf>,
            target: xc_core::PublicationTarget,
            owner: String,
            journal: PathBuf,
        },
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Replacement {
        original: PathBuf,
        repaired: PathBuf,
    }
    #[derive(Serialize, Deserialize)]
    struct RepairRecord {
        original_manifest_digest: ContentDigest,
        original_semantic_digest: ContentDigest,
        original_content_digest: ContentDigest,
        repaired_manifest_digest: ContentDigest,
        repaired_semantic_digest: ContentDigest,
        repaired_content_digest: ContentDigest,
        original_visibility: CacheVisibility,
        method: String,
    }
    fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        Ok(())
    }
    fn canonical(source: &LocalShardJson) -> Result<CanonicalArtifactManifest> {
        Ok(serde_json::from_str(
            source
                .manifest
                .tags
                .get(REMOTE_CANONICAL_MANIFEST_TAG)
                .context("canonical binding missing")?,
        )?)
    }

    fn replace_embedded_payload(
        measurement: &serde_json::Value,
        original: &serde_json::Value,
        replacement: serde_json::Value,
    ) -> Result<serde_json::Value> {
        if measurement == original {
            return Ok(replacement);
        }
        // RetainedCcmRun stores observed source payloads in an array. Preserve
        // that container and every unselected entry; never substitute by kind
        // alone or discard a wrapper while repairing its evidence digest.
        if let Some(values) = measurement.as_array() {
            let positions = values
                .iter()
                .enumerate()
                .filter_map(|(index, value)| (value == original).then_some(index))
                .collect::<Vec<_>>();
            if positions.len() == 1 {
                let mut result = values.clone();
                result[positions[0]] = replacement;
                return Ok(serde_json::Value::Array(result));
            }
        }
        bail!("capture embedded measurement does not uniquely match its original source");
    }
    fn stage(
        original: &LocalShardJson,
        bytes: &[u8],
        semantic: SemanticKeyEnvelope,
        dependencies: Vec<PayloadDependencyIdentity>,
        logical: String,
        output: &Path,
    ) -> Result<()> {
        let old = canonical(original)?;
        let mut manifest = old.clone();
        manifest.semantic_digest = semantic.digest()?;
        manifest.resolved_mathematical_configuration_digest =
            ContentDigest(xc_core::research_digest(&semantic.resolved_mathematical_parameters)?.0);
        manifest.semantic_key = semantic;
        manifest.canonical_payload.dependencies = dependencies;
        manifest.canonical_payload.ordered_items[0].content_digest = ContentDigest::sha256(bytes);
        manifest.canonical_payload.ordered_items[0].size_bytes = bytes.len() as u64;
        manifest.payload_digest = manifest.canonical_payload.digest()?;
        manifest.producer_toolkit_version = ToolkitVersion::parse(env!("CARGO_PKG_VERSION"))?;
        manifest.requested_assurance = xc_core::AssuranceLevel::Computed;
        manifest.claim_scope = "validated typed numerical cache artifact".into();
        // Visibility annotations are preserved; no old certification is carried
        // over as assurance for newly calculated values.
        let mut runtime = original.manifest.clone();
        runtime.key.parameters_digest = manifest.semantic_digest.clone();
        runtime.key.logical_key = logical.clone();
        runtime.content_digest = ContentDigest::sha256(bytes);
        runtime.size_bytes = bytes.len() as u64;
        runtime.objects = vec![CacheObjectRef {
            content_digest: runtime.content_digest.clone(),
            size_bytes: runtime.size_bytes,
        }];
        runtime.quality = CacheQuality::Validated;
        runtime.producer_toolkit_version = manifest.producer_toolkit_version.clone();
        runtime.tags.remove(REMOTE_CANONICAL_MANIFEST_TAG);
        runtime.tags.insert(
            SEMANTIC_KEY_MANIFEST_TAG.into(),
            serde_json::to_string(&manifest.semantic_key)?,
        );
        runtime.provenance_digest = Some(original.canonical_manifest_digest.clone());
        fs::create_dir(output)?;
        let output = output.canonicalize()?;
        write_new(&output.join("payload.json"), bytes)?;
        let parts = output.join("parts");
        fs::create_dir(&parts)?;
        let resources = ResourcePolicy::default();
        let cancel = CancellationToken::new();
        let zip = output.join("payload.zip");
        let package = package_canonical_payload_bytes_zip64(
            &manifest.canonical_payload,
            "payload.json",
            bytes,
            &zip,
            &resources,
            &cancel,
        )?;
        let encoding = stream_split_encoded(
            &mut fs::File::open(&zip)?,
            manifest.payload_digest.clone(),
            package.encoder_profile,
            &TransportPolicy::default(),
            &resources,
            &cancel,
            |part, bytes| {
                let path = parts.join(&part.repository_path);
                fs::create_dir_all(path.parent().expect("part parent"))?;
                let mut file = fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(path)?;
                file.write_all(bytes)?;
                file.sync_all()?;
                Ok(())
            },
        )?;
        verify_canonical_payload_zip64(&manifest.canonical_payload, &encoding, &zip, &cancel)?;
        manifest.transport_digests = vec![encoding.digest()?];
        manifest.validate()?;
        let record = RepairRecord {
            original_manifest_digest: original.canonical_manifest_digest.clone(),
            original_semantic_digest: old.semantic_digest,
            original_content_digest: original.manifest.content_digest.clone(),
            repaired_manifest_digest: manifest.digest()?,
            repaired_semantic_digest: manifest.semantic_digest.clone(),
            repaired_content_digest: runtime.content_digest.clone(),
            original_visibility: original.manifest.visibility,
            method: "retained_l2_tangent_root_response_v3_no_claim_rerun".into(),
        };
        let draft = CanonicalProductionDraft {
            schema_version: 1,
            family: manifest.artifact_family.clone(),
            source_operation: "ccm.response.retained_repair".into(),
            source_logical_key: logical,
            source_artifact_key: runtime.key.clone(),
            source_content_digest: runtime.content_digest.clone(),
            source_manifest_digest: ContentDigest(xc_core::research_digest(&runtime)?.0),
            source_quality: Some(CacheQuality::Validated),
            manifest,
            encoding,
            staged_parts_root: parts,
            achieved_assurance: ArtifactAssuranceState::Computed,
            required_assurance: None,
            assurance_evidence_digests: vec![],
        };
        write_new(
            &output.join("repair.json"),
            &serde_json::to_vec_pretty(&record)?,
        )?;
        write_new(
            &output.join("draft.json"),
            &serde_json::to_vec_pretty(&draft)?,
        )?;
        fs::remove_file(zip)?;
        Ok(())
    }
    fn response_logical(semantic: &SemanticKeyEnvelope) -> Result<String> {
        let p = &semantic.resolved_mathematical_parameters;
        let family = if semantic.artifact_kind == "ccm_u_flow_response_analysis" {
            "u-flow-response"
        } else {
            "prime-power-response"
        };
        let parity = match p.get("parity_policy").and_then(|p| p.as_str()) {
            Some("adaptive-even") => "adaptive-even",
            Some("natural") => "natural",
            Some("even-sector") => "even",
            Some(_) => bail!("unsupported retained parity policy"),
            None => {
                if p["force_even"] == true {
                    "even"
                } else {
                    "natural"
                }
            }
        };
        Ok(format!(
            "ccm/{family}/{}/{}/{}/{parity}/{}",
            p["lambda_squared"].as_str().context("lambda identity")?,
            p["n_modes"],
            p["precision_bits"],
            semantic.digest()?
        ))
    }
    pub fn run() -> Result<()> {
        let path = std::env::args()
            .nth(1)
            .context("usage: ccm_response_repair REQUEST.json")?;
        let request: Request = serde_json::from_slice(&fs::read(path)?)?;
        let start = std::time::Instant::now();
        match request {
            Request::Response {
                response,
                eigenpair,
                output,
                read,
            } => {
                let original = read_local_shard_json(&response, &read)?;
                let state = read_local_shard_json(&eigenpair, &read)?;
                let repaired = repair_retained_response(&original, &state)?;
                let logical = response_logical(&repaired.semantic_key)?;
                let dependencies = canonical(&original)?.canonical_payload.dependencies;
                stage(
                    &original,
                    &repaired.payload,
                    repaired.semantic_key,
                    dependencies,
                    logical,
                    &output,
                )?;
            }
            Request::Receipt {
                receipt,
                repairs,
                output,
                read,
            } => {
                let original = read_local_shard_json(&receipt, &read)?;
                if original.manifest.key.kind != CAPTURE_RECEIPT_KIND
                    || original.manifest.visibility == CacheVisibility::Public
                {
                    bail!("capture repair requires a private capture receipt");
                }
                let record: CaptureArtifact = serde_json::from_slice(&original.payload)?;
                record.validate()?;
                let mut updates = Vec::new();
                let mut dependencies = canonical(&original)?.canonical_payload.dependencies;
                for replacement in repairs {
                    let source = read_local_shard_json(&replacement.original, &read)?;
                    let old = canonical(&source)?;
                    let draft: CanonicalProductionDraft = serde_json::from_slice(&fs::read(
                        replacement.repaired.join("draft.json"),
                    )?)?;
                    draft.manifest.validate()?;
                    let repair_record: RepairRecord = serde_json::from_slice(&fs::read(
                        replacement.repaired.join("repair.json"),
                    )?)?;
                    let bytes = fs::read(replacement.repaired.join("payload.json"))?;
                    if repair_record.original_manifest_digest != source.canonical_manifest_digest
                        || repair_record.repaired_manifest_digest != draft.manifest.digest()?
                        || ContentDigest::sha256(&bytes) != draft.source_content_digest
                    {
                        bail!("capture replacement source binding mismatch");
                    }
                    let id = match old.semantic_key.artifact_kind.as_str() {
                        "ccm_prime_power_response_analysis" => "prime_power_response",
                        "ccm_u_flow_response_analysis" => "u_flow_response",
                        _ => bail!("unsupported capture replacement kind"),
                    };
                    let measurement = record
                        .measurements
                        .get(id)
                        .context("capture measurement missing")?;
                    let replacement_value = replace_embedded_payload(
                        &measurement.value,
                        &serde_json::from_slice::<serde_json::Value>(&source.payload)?,
                        serde_json::from_slice(&bytes)?,
                    )?;
                    let mut replacement_dependencies = measurement.source_dependencies.clone();
                    let mut matched = 0;
                    for dep in &mut replacement_dependencies {
                        if dep.key.kind == source.manifest.key.kind
                            && dep.key.parameters_digest == old.semantic_digest
                            && dep.content_digest == source.manifest.content_digest
                        {
                            let old_suffix = format!("/{}", old.semantic_digest);
                            let base = dep
                                .key
                                .logical_key
                                .strip_suffix(&old_suffix)
                                .context("unexpected original response logical key")?;
                            dep.key.logical_key =
                                format!("{base}/{}", draft.manifest.semantic_digest);
                            dep.key.parameters_digest = draft.manifest.semantic_digest.clone();
                            dep.content_digest = draft.source_content_digest.clone();
                            matched += 1;
                        }
                    }
                    if matched != 1 {
                        bail!("capture response dependency not uniquely bound");
                    }
                    updates.push(CaptureMeasurementRepair {
                        diagnostic: id.into(),
                        original_evidence_digest: xc_core::research_digest(measurement)?.0,
                        replacement: CapturedMeasurement {
                            value: replacement_value,
                            source_dependencies: replacement_dependencies,
                        },
                    });
                    let mut canonical_matches = 0;
                    for dep in &mut dependencies {
                        if dep.manifest_digest == source.canonical_manifest_digest
                            && dep.semantic_digest == old.semantic_digest
                            && dep.payload_digest == old.payload_digest
                        {
                            dep.manifest_digest = draft.manifest.digest()?;
                            dep.semantic_digest = draft.manifest.semantic_digest.clone();
                            dep.payload_digest = draft.manifest.payload_digest.clone();
                            canonical_matches += 1;
                        }
                    }
                    if canonical_matches != 1 {
                        bail!("capture canonical dependency not uniquely bound");
                    }
                }
                if updates.is_empty() {
                    bail!("no capture measurements selected for repair");
                }
                let repaired = repair_capture_measurements(&record, updates)?;
                dependencies.sort_by(|a, b| {
                    (
                        &a.artifact_family,
                        &a.semantic_digest,
                        &a.manifest_digest,
                        &a.payload_digest,
                    )
                        .cmp(&(
                            &b.artifact_family,
                            &b.semantic_digest,
                            &b.manifest_digest,
                            &b.payload_digest,
                        ))
                });
                let digest = xc_core::research_digest(&repaired)?;
                let mut semantic = canonical(&original)?.semantic_key;
                semantic.resolved_mathematical_parameters = serde_json::json!({"record_digest": digest, "source_dependencies": repaired.source_dependencies});
                let logical = format!(
                    "{}attempt/{}",
                    capture_receipt_plan_prefix(repaired.receipt.plan_digest())?,
                    digest.0
                );
                stage(
                    &original,
                    &serde_json::to_vec(&repaired)?,
                    semantic,
                    dependencies,
                    logical,
                    &output,
                )?;
            }
            Request::Publish {
                drafts,
                target,
                owner,
                journal,
            } => {
                if !matches!(
                    target,
                    xc_core::PublicationTarget::Public | xc_core::PublicationTarget::Private
                ) {
                    bail!("repair publication must select one original visibility lane");
                }
                let drafts = drafts
                    .iter()
                    .map(|path| -> Result<_> {
                        let record: RepairRecord = serde_json::from_slice(&fs::read(
                            path.parent().context("draft parent")?.join("repair.json"),
                        )?)?;
                        let is_public = record.original_visibility == CacheVisibility::Public;
                        if is_public != (target == xc_core::PublicationTarget::Public) {
                            bail!("repair publication cannot change source visibility");
                        }
                        let draft: CanonicalProductionDraft =
                            serde_json::from_slice(&fs::read(path)?)?;
                        if record.repaired_manifest_digest != draft.manifest.digest()? {
                            bail!("repair draft binding mismatch");
                        }
                        Ok(draft)
                    })
                    .collect::<Result<Vec<_>>>()?;
                GitHubCredentialApiProbe::default().prepare_git_transport()?;
                let report = execute_managed_drafts_on_github(
                    &drafts,
                    target,
                    &owner,
                    &journal,
                    &ResourcePolicy::default(),
                    false,
                )?;
                println!("{}", serde_json::to_string_pretty(&report)?);
                if !report.all_completed {
                    bail!(
                        "retained repair publication is incomplete; preserve the journal for retry"
                    );
                }
            }
        }
        eprintln!(
            "PASS: retained artifact operation completed in {:.3} s",
            start.elapsed().as_secs_f64()
        );
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use serde_json::json;

        #[test]
        fn embedded_repair_preserves_retained_run_array_and_requires_exact_source() {
            let original = json!({"root_velocity_responses":["123"]});
            let repaired = json!({"root_velocity_responses":["4"]});
            let other = json!({"unrelated":"preserve"});
            assert_eq!(
                replace_embedded_payload(&original, &original, repaired.clone()).unwrap(),
                repaired
            );
            assert_eq!(
                replace_embedded_payload(&json!([original.clone()]), &original, repaired.clone())
                    .unwrap(),
                json!([repaired.clone()])
            );
            assert_eq!(
                replace_embedded_payload(
                    &json!([other.clone(), original.clone()]),
                    &original,
                    repaired.clone()
                )
                .unwrap(),
                json!([other, repaired.clone()])
            );
            assert!(replace_embedded_payload(
                &json!([original.clone(), original.clone()]),
                &original,
                repaired.clone()
            )
            .is_err());
            assert!(replace_embedded_payload(
                &json!([{"root_velocity_responses":["999"]}]),
                &original,
                repaired
            )
            .is_err());
        }
    }
}
#[cfg(feature = "hp")]
fn main() -> anyhow::Result<()> {
    enabled::run()
}
#[cfg(not(feature = "hp"))]
fn main() {
    eprintln!("ccm_response_repair requires --features hp");
    std::process::exit(2);
}
