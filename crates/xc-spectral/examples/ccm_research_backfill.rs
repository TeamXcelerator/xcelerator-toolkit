//! Offline, additive, resumable retained-source batch. No primary solver or publication.
#[cfg(feature = "hp")]
mod app {
    use anyhow::{bail, Context, Result};
    use serde::{Deserialize, Serialize};
    use serde_json::{json, Value};
    use std::{
        io::Write,
        path::{Path, PathBuf},
    };
    use xc_cache::*;
    use xc_spectral::ccm::{retained_evidence::*, state_geometry::*};
    #[derive(Clone, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Files {
        manifest: PathBuf,
        payload: PathBuf,
        maximum_payload_bytes: Option<u64>,
    }
    #[derive(Serialize, Deserialize)]
    #[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
    enum Task {
        ExtendedResearch {
            diagnostic: String,
            state: Files,
            matrix: Option<Files>,
            roots: Option<Files>,
            secular: Option<Files>,
            #[serde(default)]
            parent_manifests: Vec<PathBuf>,
            input: Option<PathBuf>,
            input_sha256: Option<ContentDigest>,
            options: Option<xc_spectral::ccm::extended_research::ExtensionOptions>,
        },
        StateGeometry {
            state: Files,
            options: Option<GeometryOptions>,
        },
        OperatorEnergy {
            state: Files,
            matrix: Files,
            #[serde(default)]
            parent_manifests: Vec<PathBuf>,
        },
        IndexedTransform {
            state: Files,
            roots: Files,
            secular: Files,
            options: Option<TransformOptions>,
        },
        IndexedDatasetTransform {
            state: Files,
            dataset: Files,
            options: Option<TransformOptions>,
        },
        RootWindow {
            state: Files,
            roots: Files,
            secular: Files,
        },
        ReferenceSource {
            spec: ReferenceSpec,
        },
        ReferenceDataset {
            spec: DatasetSpec,
        },
        ReferenceProjection {
            state: Files,
            inputs: ResearchInputs,
        },
        ObservationPacket {
            observation: ExternalObservation,
        },
        Stabilization {
            states: Vec<Files>,
            options: StabilizationOptions,
        },
    }
    #[derive(Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Job {
        id: String,
        task: Task,
    }
    #[derive(Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Batch {
        schema_version: u32,
        approved_payload_digests: Vec<ContentDigest>,
        cache_root: PathBuf,
        jobs: Vec<Job>,
    }
    fn read(base: &Path, files: &Files) -> Result<(ArtifactManifest, Vec<u8>)> {
        if std::fs::metadata(base.join(&files.manifest))?.len() > 1024 * 1024 {
            bail!("manifest exceeds byte budget");
        }
        let manifest: ArtifactManifest =
            serde_json::from_slice(&std::fs::read(base.join(&files.manifest))?)?;
        let limit = files.maximum_payload_bytes.unwrap_or(512 * 1024 * 1024);
        if manifest.size_bytes > limit
            || std::fs::metadata(base.join(&files.payload))?.len() > limit
        {
            bail!("source payload exceeds declared byte budget");
        }
        let bytes = std::fs::read(base.join(&files.payload))?;
        Ok((manifest, bytes))
    }
    fn state(base: &Path, f: &Files, approved: &[ContentDigest]) -> Result<RetainedState> {
        let (m, b) = read(base, f)?;
        RetainedState::from_payload(&m, &b, approved)
    }
    fn packet<T: Serialize>(r: ArtifactExecutionCacheResult<T>) -> Result<Value> {
        let m = r
            .produced_manifest
            .or(r.reused_manifest)
            .context("managed manifest missing")?;
        Ok(json!({"manifest":m,"report":r.value}))
    }
    fn execute(
        job: &Job,
        base: &Path,
        approved: &[ContentDigest],
        cache: &ArtifactCacheContext<'_>,
    ) -> Result<Value> {
        match &job.task {
            Task::ExtendedResearch {
                diagnostic,
                state: f,
                matrix,
                roots,
                secular,
                parent_manifests,
                input,
                input_sha256,
                options,
            } => {
                use xc_spectral::ccm::extended_research::*;
                let s = state(base, f, approved)?;
                let m = matrix
                    .as_ref()
                    .map(|f| {
                        let (m, b) = read(base, f)?;
                        RetainedMatrix::from_payload(&m, &b, approved)
                    })
                    .transpose()?;
                let roots = match (roots, secular) {
                    (Some(r), Some(sec)) => {
                        let (r, b) = read(base, r)?;
                        let (m, mb) = read(base, sec)?;
                        Some(RetainedRoots::from_payload(&r, &b, &m, &mb, &s, approved)?)
                    }
                    (None, None) => None,
                    _ => bail!("root window and secular source must be supplied together"),
                };
                if input.is_some() != input_sha256.is_some() {
                    bail!("external input path and SHA-256 must be frozen together");
                }
                let inputs = input
                    .as_ref()
                    .zip(input_sha256.as_ref())
                    .map(|(path, expected)| -> Result<_> {
                        use std::io::Read;
                        let path = base.join(path);
                        let mut bytes = Vec::new();
                        std::fs::File::open(&path)?
                            .take(64 * 1024 * 1024 + 1)
                            .read_to_end(&mut bytes)?;
                        if bytes.len() > 64 * 1024 * 1024 {
                            bail!("external input exceeds byte budget");
                        }
                        if ContentDigest::sha256(&bytes) != *expected {
                            bail!("external input does not match frozen batch digest");
                        }
                        let value: serde_json::Value = serde_json::from_slice(&bytes)?;
                        if value.get("source_eigenpair").is_some() {
                            ExternalResearchInputs::from_bytes(&path, &bytes)
                        } else {
                            xc_spectral::ccm::research_completion::ReferencePreparation::from_bytes(
                                &path, &bytes,
                            )?
                            .prepare(&s, roots.as_ref())
                        }
                    })
                    .transpose()?;
                let parents = parent_manifests
                    .iter()
                    .map(|p| -> Result<ArtifactManifest> {
                        if std::fs::metadata(base.join(p))?.len() > 1024 * 1024 {
                            bail!("ancestry manifest exceeds byte budget");
                        }
                        let m: ArtifactManifest =
                            serde_json::from_slice(&std::fs::read(base.join(p))?)?;
                        if !approved.contains(&m.content_digest) {
                            bail!("unapproved ancestry manifest");
                        }
                        Ok(m)
                    })
                    .collect::<Result<Vec<_>>>()?;
                if parents.len() > 64 {
                    bail!("ancestry manifest count exceeded");
                }
                let mut o = options
                    .clone()
                    .unwrap_or_else(|| ExtensionOptions::for_source(&s));
                if options.is_none() {
                    o.working_precision_bits = o.working_precision_bits.max(
                        inputs
                            .as_ref()
                            .map_or(0, |i| i.precision_bits.saturating_add(64)),
                    );
                }
                if options.is_none() {
                    if let Some(r) = &roots {
                        o.working_precision_bits = o
                            .working_precision_bits
                            .max(TransformOptions::for_roots(&s, r).working_precision_bits);
                    }
                }
                if diagnostic == "external_source" {
                    return packet(capture_external_source(
                        &s,
                        inputs.as_ref().context("external source input required")?,
                        cache,
                    )?);
                }
                packet(capture_extended(
                    diagnostic,
                    &s,
                    m.as_ref(),
                    roots.as_ref(),
                    inputs.as_ref(),
                    &o,
                    &parents,
                    cache,
                )?)
            }
            Task::StateGeometry { state: f, options } => {
                let s = state(base, f, approved)?;
                packet(analyze_state_geometry_via_cache(
                    &s,
                    &options
                        .clone()
                        .unwrap_or_else(|| GeometryOptions::for_source(&s)),
                    cache,
                )?)
            }
            Task::OperatorEnergy {
                state: f,
                matrix,
                parent_manifests,
            } => {
                let s = state(base, f, approved)?;
                let (m, b) = read(base, matrix)?;
                let m = RetainedMatrix::from_payload(&m, &b, approved)?;
                if parent_manifests.len() > 64 {
                    bail!("ancestry manifest budget exceeded");
                }
                let parents = parent_manifests
                    .iter()
                    .map(|p| -> Result<ArtifactManifest> {
                        if std::fs::metadata(base.join(p))?.len() > 1024 * 1024 {
                            bail!("manifest exceeds byte budget");
                        }
                        Ok(serde_json::from_slice(&std::fs::read(base.join(p))?)?)
                    })
                    .collect::<Result<Vec<_>>>()?;
                if parents
                    .iter()
                    .any(|m| !approved.contains(&m.content_digest))
                {
                    bail!("unapproved ancestry manifest");
                }
                packet(capture_operator_energy_with_ancestry(
                    &s, &m, &parents, cache,
                )?)
            }
            Task::IndexedTransform {
                state: f,
                roots,
                secular,
                options,
            } => {
                let s = state(base, f, approved)?;
                let (r, b) = read(base, roots)?;
                let (m, mb) = read(base, secular)?;
                let r = RetainedRoots::from_payload(&r, &b, &m, &mb, &s, approved)?;
                packet(capture_transforms_at_roots(
                    &s,
                    &r,
                    &options
                        .clone()
                        .unwrap_or_else(|| TransformOptions::for_roots(&s, &r)),
                    cache,
                )?)
            }
            Task::IndexedDatasetTransform {
                state: f,
                dataset,
                options,
            } => {
                let s = state(base, f, approved)?;
                let (m, b) = read(base, dataset)?;
                let d = RetainedDataset::from_payload(&m, &b, approved)?;
                packet(capture_transforms_at_dataset(
                    &s,
                    &d,
                    &options
                        .clone()
                        .unwrap_or_else(|| TransformOptions::for_dataset(&s, &d)),
                    cache,
                )?)
            }
            Task::RootWindow {
                state: f,
                roots,
                secular,
            } => {
                let s = state(base, f, approved)?;
                let (r, b) = read(base, roots)?;
                let (m, mb) = read(base, secular)?;
                let r = RetainedRoots::from_payload(&r, &b, &m, &mb, &s, approved)?;
                packet(capture_root_window(&r, cache)?)
            }
            Task::ReferenceSource { spec } => packet(capture_reference(spec, cache)?),
            Task::ReferenceDataset { spec } => packet(capture_dataset(spec, cache)?),
            Task::ReferenceProjection { state: f, inputs } => {
                let s = state(base, f, approved)?;
                let r = capture_configured_projection(&s, inputs, cache)?;
                if r.sources.len() != 1 {
                    bail!("projection manifest unavailable");
                }
                Ok(json!({"manifest":r.sources[0],"report":r.value}))
            }
            Task::ObservationPacket { observation } => {
                packet(capture_external_observation(observation, cache)?)
            }
            Task::Stabilization { states, options } => {
                let s = states
                    .iter()
                    .map(|f| state(base, f, approved))
                    .collect::<Result<Vec<_>>>()?;
                packet(capture_stabilization(&s, options, cache)?)
            }
        }
    }
    fn save_new(path: &Path, bytes: &[u8]) -> Result<()> {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let temp = path.with_extension(format!("pending-{}-{stamp}", std::process::id()));
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
        drop(f);
        // Creating a hard link is atomic and refuses an existing destination.
        let result = std::fs::hard_link(&temp, path);
        let _ = std::fs::remove_file(&temp);
        result?;
        Ok(())
    }
    pub fn run() -> Result<()> {
        let args = std::env::args().skip(1).collect::<Vec<_>>();
        if args.len() != 2 {
            bail!("usage: ccm_research_backfill BATCH.json OUTPUT_DIRECTORY");
        }
        let input = PathBuf::from(&args[0]).canonicalize()?;
        let base = input.parent().context("batch has no parent")?;
        if std::fs::metadata(&input)?.len() > 32 * 1024 * 1024 {
            bail!("batch exceeds 32 MiB");
        }
        let bytes = std::fs::read(&input)?;
        let batch: Batch = serde_json::from_slice(&bytes)?;
        if batch.schema_version != 1 || batch.jobs.is_empty() || batch.jobs.len() > 10000 {
            bail!("unsupported batch");
        }
        let mut ids = std::collections::BTreeSet::new();
        for j in &batch.jobs {
            if j.id.is_empty()
                || j.id.len() > 100
                || !j
                    .id
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
                || !ids.insert(j.id.clone())
            {
                bail!("invalid or repeated job ID");
            }
        }
        let output = PathBuf::from(&args[1]);
        std::fs::create_dir_all(&output)?;
        let seal = ContentDigest::sha256(&serde_json::to_vec(&batch)?);
        let seal_file = output.join("batch.sha256");
        if seal_file.exists() {
            if std::fs::read_to_string(&seal_file)? != seal.0 {
                bail!("output belongs to a different frozen batch");
            }
        } else {
            save_new(&seal_file, seal.0.as_bytes())?;
        }
        let resolver = CacheResolver::new(vec![CacheLayer {
            precedence: 0,
            store: Box::new(FilesystemCacheStore::new(
                "retained-backfill",
                base.join(&batch.cache_root),
                true,
                CacheVisibility::Local,
            )),
        }]);
        let policy = CachePolicy {
            current_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION"))?,
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
            ordered_overlays: vec!["retained-backfill".into()],
            mode: ArtifactExecutionCacheMode::PreferReuse,
            write_on_miss: true,
            write_visibility: CacheVisibility::Local,
            requested_assurance: xc_core::AssuranceLevel::Computed,
            certification_failure_policy: CertificationFailurePolicy::RetainComputedFailRun,
            production_sink: None,
        };
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let mut outcomes = Vec::new();
        let mut failures = 0;
        for job in &batch.jobs {
            let result = xc_numerics::hp_runtime::run_hp(|| {
                execute(job, base, &batch.approved_payload_digests, &cache)
            })
            .and_then(|value| {
                let path = output.join(format!("{}.json", job.id));
                if path.exists() {
                    let old: Value = serde_json::from_slice(&std::fs::read(&path)?)?;
                    if old["manifest"]["key"] != value["manifest"]["key"]
                        || old["manifest"]["content_digest"] != value["manifest"]["content_digest"]
                        || old["report"] != value["report"]
                    {
                        bail!("retained output mismatch; not overwritten");
                    }
                } else {
                    save_new(&path, &serde_json::to_vec_pretty(&value)?)?;
                }
                Ok(value)
            });
            match result {
                Ok(value) => {
                    let mut row_outcomes = std::collections::BTreeMap::<String, usize>::new();
                    if let Some(rows) = value["report"]["data"]["rows"].as_array() {
                        for row in rows {
                            let status = row["outcome"]
                                .as_str()
                                .or_else(|| row["status"].as_str())
                                .unwrap_or("unassessed");
                            *row_outcomes.entry(status.to_string()).or_default() += 1;
                        }
                    }
                    let outcome = value["report"]["data"]["outcome"]
                        .as_str()
                        .unwrap_or("see_report_assurance_and_row_outcomes");
                    println!("[RETAINED] {}: {}", job.id, outcome);
                    outcomes.push(json!({"id":job.id,"status":"retained","numerical_outcome":outcome,"row_outcomes":row_outcomes,"numerical_coverage":xc_cache::NumericalCoverage::from_value(&value["report"]),"child":value["manifest"]["content_digest"]}));
                }
                Err(e) => {
                    failures += 1;
                    let failure = CaptureFailure::failed(e);
                    eprintln!("[INCOMPLETE] {}", job.id);
                    outcomes.push(json!({"id":job.id,"failure":failure}));
                }
            }
            // Append-only attempts: a successful later retry never rewrites an old failure.
            save_new(
                &output.join(format!("attempt-{stamp}-{}.json", job.id)),
                &serde_json::to_vec_pretty(outcomes.last().unwrap())?,
            )?;
        }
        save_new(
            &output.join(format!("summary-{stamp}.json")),
            &serde_json::to_vec_pretty(
                &json!({"batch":seal,"publication":"not_requested","failed_jobs":failures,"outcomes":outcomes}),
            )?,
        )?;
        if failures > 0 {
            bail!("{failures} jobs incomplete; successful children preserved; rerun the same frozen batch to retry");
        }
        Ok(())
    }
}
#[cfg(feature = "hp")]
fn main() -> anyhow::Result<()> {
    app::run()
}
#[cfg(not(feature = "hp"))]
fn main() {
    eprintln!("ccm_research_backfill requires --features hp");
    std::process::exit(2);
}
