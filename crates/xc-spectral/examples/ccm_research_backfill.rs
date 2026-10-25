//! Offline, additive, resumable retained-source batch. No primary solver or publication.
#[cfg(feature = "hp")]
mod app {
    use anyhow::{bail, Context, Result};
    use serde::{Deserialize, Serialize};
    use serde_json::{json, Value};
    use std::{
        cell::RefCell,
        io::{BufReader, Read, Write},
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
    // Keep conservative ownership allowances through the whole job, including
    // decoded states, roots and input objects after their JSON buffers drop.
    struct Sources<'a> {
        base: &'a Path,
        resources: &'a xc_core::RunResources,
        reservations: RefCell<Vec<xc_core::MemoryReservation>>,
    }
    impl Sources<'_> {
        fn manifest(&self, files: &Files) -> Result<ArtifactManifest> {
            let path = self.base.join(&files.manifest);
            let mut bytes = Vec::new();
            std::fs::File::open(path)?
                .take(1024 * 1024 + 1)
                .read_to_end(&mut bytes)?;
            if bytes.len() > 1024 * 1024 {
                bail!("manifest exceeds byte budget");
            }
            let manifest: ArtifactManifest = serde_json::from_slice(&bytes)?;
            manifest.validate()?;
            if files
                .maximum_payload_bytes
                .is_some_and(|limit| manifest.size_bytes > limit)
            {
                bail!("source payload exceeds declared byte budget");
            }
            Ok(manifest)
        }
        fn reserve_json(&self, size: u64) -> Result<()> {
            // Match capture_runtime's conservative retained-JSON allowance.
            let owned = size
                .checked_mul(64)
                .and_then(|n| n.checked_add(4096))
                .context("retained JSON ownership estimate overflow")?;
            let reservation = self.resources.try_reserve_retained(owned)?;
            self.reservations.borrow_mut().push(reservation);
            Ok(())
        }
        fn read_bytes(&self, path: &Path, expected: u64) -> Result<Vec<u8>> {
            let mut file = std::fs::File::open(path)?;
            if file.metadata()?.len() != expected {
                bail!("source payload length differs from declared size");
            }
            self.reserve_json(expected)?;
            let length =
                usize::try_from(expected).context("source payload exceeds address space")?;
            let mut bytes = Vec::new();
            bytes
                .try_reserve_exact(length)
                .context("source payload allocation failed")?;
            bytes.resize(length, 0);
            file.read_exact(&mut bytes)?;
            if file.read(&mut [0u8; 1])? != 0 {
                bail!("source payload grew beyond declared size");
            }
            Ok(bytes)
        }
        fn read(&self, files: &Files) -> Result<(ArtifactManifest, Vec<u8>)> {
            let manifest = self.manifest(files)?;
            let bytes = self.read_bytes(&self.base.join(&files.payload), manifest.size_bytes)?;
            Ok((manifest, bytes))
        }
        fn state(&self, files: &Files, approved: &[ContentDigest]) -> Result<RetainedState> {
            let (manifest, bytes) = self.read(files)?;
            #[derive(Deserialize)]
            struct Shape {
                n_modes: usize,
                precision_bits: u32,
            }
            let shape: Shape = serde_json::from_slice(&bytes)?;
            if shape.n_modes > 8192 || !(64..=1_000_000).contains(&shape.precision_bits) {
                bail!("unsupported retained state shape or precision");
            }
            // Short decimal strings can expand into high-precision Float
            // allocations; account for those independently of JSON storage.
            let numeric_bytes = (2 * shape.n_modes as u64 + 1)
                .checked_mul(u64::from(shape.precision_bits).div_ceil(8) + 96)
                .context("retained state ownership estimate overflow")?;
            self.reservations
                .borrow_mut()
                .push(self.resources.try_reserve_retained(numeric_bytes)?);
            RetainedState::from_payload(&manifest, &bytes, approved)
        }
        fn matrix(
            &self,
            files: &Files,
            approved: &[ContentDigest],
        ) -> Result<RetainedMatrix<'static>> {
            let manifest = self.manifest(files)?;
            let file = std::fs::File::open(self.base.join(&files.payload))?;
            if file.metadata()?.len() != manifest.size_bytes {
                bail!("source payload length differs from declared size");
            }
            let mut reader = BufReader::new(file);
            RetainedMatrix::from_reader_with_resources(
                &manifest,
                &mut reader,
                approved,
                self.resources,
            )
        }
        fn external_bytes(&self, path: &Path) -> Result<Vec<u8>> {
            let size = std::fs::metadata(path)?.len();
            // This is the input schema's independent domain guard. The job's
            // configured working budget controls ownership below that guard.
            if size > xc_spectral::ccm::capture_runtime::RESEARCH_INPUT_MAXIMUM_BYTES {
                bail!("external input exceeds supported input domain");
            }
            self.read_bytes(path, size)
        }
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
        resources: &xc_core::RunResources,
    ) -> Result<Value> {
        let sources = Sources {
            base,
            resources,
            reservations: RefCell::new(Vec::new()),
        };
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
                let s = sources.state(f, approved)?;
                let m = matrix
                    .as_ref()
                    .map(|f| sources.matrix(f, approved))
                    .transpose()?;
                let roots = match (roots, secular) {
                    (Some(r), Some(sec)) => {
                        let (r, b) = sources.read(r)?;
                        let (m, mb) = sources.read(sec)?;
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
                        let path = base.join(path);
                        let bytes = sources.external_bytes(&path)?;
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
                let s = sources.state(f, approved)?;
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
                let s = sources.state(f, approved)?;
                let m = sources.matrix(matrix, approved)?;
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
                let s = sources.state(f, approved)?;
                let (r, b) = sources.read(roots)?;
                let (m, mb) = sources.read(secular)?;
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
                let s = sources.state(f, approved)?;
                let (m, b) = sources.read(dataset)?;
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
                let s = sources.state(f, approved)?;
                let (r, b) = sources.read(roots)?;
                let (m, mb) = sources.read(secular)?;
                let r = RetainedRoots::from_payload(&r, &b, &m, &mb, &s, approved)?;
                packet(capture_root_window(&r, cache)?)
            }
            Task::ReferenceSource { spec } => packet(capture_reference(spec, cache)?),
            Task::ReferenceDataset { spec } => packet(capture_dataset(spec, cache)?),
            Task::ReferenceProjection { state: f, inputs } => {
                let s = sources.state(f, approved)?;
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
                    .map(|f| sources.state(f, approved))
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
        let resource_policy =
            xc_spectral::ccm::capture_runtime::CaptureResourcePolicy::from_environment()?;
        let resources = xc_core::RunResources::current().map_or_else(
            || {
                xc_core::RunResources::with_worker_limit(
                    resource_policy.maximum_working_bytes,
                    rayon::current_num_threads(),
                )
            },
            Ok,
        )?;
        let _resource_scope = resources.enter();
        let mut outcomes = Vec::new();
        let mut failures = 0;
        for job in &batch.jobs {
            let result = xc_numerics::hp_runtime::run_hp(|| {
                // Keep the batch owner explicit at the HP callback boundary.
                let _job_resource_scope = resources.enter();
                execute(
                    job,
                    base,
                    &batch.approved_payload_digests,
                    &cache,
                    &resources,
                )
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

    #[cfg(test)]
    mod tests {
        use super::*;
        fn sources<'a>(base: &'a Path, resources: &'a xc_core::RunResources) -> Sources<'a> {
            Sources {
                base,
                resources,
                reservations: RefCell::new(Vec::new()),
            }
        }
        fn fixture(root: &Path) -> Files {
            let store = FilesystemCacheStore::new(
                "fixture",
                root.join("cache"),
                true,
                CacheVisibility::Local,
            );
            let version = ToolkitVersion::parse(env!("CARGO_PKG_VERSION")).unwrap();
            let draft = ArtifactDraft {
                schema_version: 1,
                key: ArtifactKey::new("fixture", "backfill", b"exact").unwrap(),
                producer_toolkit_version: version.clone(),
                minimum_reader_version: version,
                maximum_reader_version: None,
                quality: CacheQuality::Validated,
                visibility: CacheVisibility::Local,
                immutable: true,
                dependencies: Vec::new(),
                tags: Default::default(),
                provenance_digest: None,
            };
            let bytes = b"{\"value\":1}";
            let manifest = store.put(&draft, bytes).unwrap();
            std::fs::write(
                root.join("manifest.json"),
                serde_json::to_vec(&manifest).unwrap(),
            )
            .unwrap();
            std::fs::write(root.join("payload.json"), bytes).unwrap();
            Files {
                manifest: "manifest.json".into(),
                payload: "payload.json".into(),
                maximum_payload_bytes: None,
            }
        }
        #[test]
        fn backfill_source_budget_refuses_retries_and_retains_job_charge() {
            let root = xc_core::test_support::TestDir::new("backfill-budget");
            let files = fixture(&root);
            let small = xc_core::RunResources::new(4096).unwrap();
            assert!(sources(&root, &small).read(&files).is_err());
            assert_eq!(small.usage().retained_bytes, 0);
            let large = xc_core::RunResources::new(16_384).unwrap();
            let owner = sources(&root, &large);
            let (manifest, bytes) = owner.read(&files).unwrap();
            assert_eq!(ContentDigest::sha256(&bytes), manifest.content_digest);
            drop(bytes);
            assert!(large.usage().retained_bytes > 0);
            drop(owner);
            assert_eq!(large.usage().retained_bytes, 0);
        }
        #[test]
        fn backfill_source_explicit_limit_and_exact_length_remain_enforced() {
            let root = xc_core::test_support::TestDir::new("backfill-limits");
            let mut files = fixture(&root);
            let resources = xc_core::RunResources::new(16_384).unwrap();
            files.maximum_payload_bytes = Some(1);
            assert!(sources(&root, &resources).read(&files).is_err());
            assert_eq!(resources.usage().retained_bytes, 0);
            files.maximum_payload_bytes = None;
            std::fs::write(root.join("payload.json"), b"truncated").unwrap();
            assert!(sources(&root, &resources).read(&files).is_err());
            assert_eq!(resources.usage().retained_bytes, 0);
        }
        #[test]
        fn backfill_source_unspecified_limit_has_no_fixed_payload_ceiling() {
            let root = xc_core::test_support::TestDir::new("backfill-large-plan");
            let mut files = fixture(&root);
            let path = root.join(&files.manifest);
            let mut manifest: ArtifactManifest =
                serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
            manifest.size_bytes = 20u64 << 30;
            manifest.objects[0].size_bytes = manifest.size_bytes;
            std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
            let resources = xc_core::RunResources::new(128u64 << 30).unwrap();
            let owner = sources(&root, &resources);
            assert_eq!(owner.manifest(&files).unwrap().size_bytes, 20u64 << 30);
            files.maximum_payload_bytes = Some(1u64 << 30);
            assert!(owner.manifest(&files).is_err());
            // Only metadata was consulted: no large file or reservation exists.
            assert_eq!(resources.usage().peak_bytes, 0);
        }
        #[test]
        fn backfill_source_matrix_streaming_retains_charge_and_authenticates() {
            let root = xc_core::test_support::TestDir::new("backfill-matrix");
            let files = fixture(&root);
            let path = root.join(&files.manifest);
            let mut manifest: ArtifactManifest =
                serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
            let bytes = br#"{"schema_version":2,"lambda_squared":"9","n_modes":0,"precision_bits":128,"entries":["1"]}"#;
            manifest.key = ArtifactKey::new("ccm_tau_matrix", "backfill-matrix", b"exact").unwrap();
            manifest.size_bytes = bytes.len() as u64;
            manifest.content_digest = ContentDigest::sha256(bytes);
            manifest.objects = vec![CacheObjectRef {
                content_digest: manifest.content_digest.clone(),
                size_bytes: manifest.size_bytes,
            }];
            std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
            std::fs::write(root.join(&files.payload), bytes).unwrap();
            let approved = std::slice::from_ref(&manifest.content_digest);
            let small = xc_core::RunResources::new(1).unwrap();
            assert!(sources(&root, &small).matrix(&files, approved).is_err());
            assert_eq!(
                small.usage().retained_bytes + small.usage().working_bytes,
                0
            );
            let resources = xc_core::RunResources::new(4 << 20).unwrap();
            let matrix = sources(&root, &resources).matrix(&files, approved).unwrap();
            assert!(resources.usage().retained_bytes > 0);
            drop(matrix);
            assert_eq!(
                resources.usage().retained_bytes + resources.usage().working_bytes,
                0
            );
            let mut corrupt = bytes.to_vec();
            let index = corrupt.iter().rposition(|b| *b == b'1').unwrap();
            corrupt[index] = b'2';
            std::fs::write(root.join(&files.payload), corrupt).unwrap();
            assert!(sources(&root, &resources).matrix(&files, approved).is_err());
            assert_eq!(
                resources.usage().retained_bytes + resources.usage().working_bytes,
                0
            );
        }
        #[test]
        fn backfill_source_external_input_uses_working_budget_and_exact_bytes() {
            let root = xc_core::test_support::TestDir::new("backfill-input");
            let path = root.join("input.json");
            std::fs::write(&path, b"{\"value\":1}").unwrap();
            let small = xc_core::RunResources::new(4096).unwrap();
            assert!(sources(&root, &small).external_bytes(&path).is_err());
            let large = xc_core::RunResources::new(16_384).unwrap();
            let owner = sources(&root, &large);
            assert_eq!(owner.external_bytes(&path).unwrap(), b"{\"value\":1}");
            assert!(large.usage().retained_bytes > 0);
        }
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
