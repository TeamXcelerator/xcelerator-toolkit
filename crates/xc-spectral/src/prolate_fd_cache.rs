// Managed finite-Dirichlet mode recovery and profile reuse. The complete source
// spectrum remains the historical artifact; no selected-only solve is introduced.
use super::*;
use xc_cache::{
    resolve_or_compute_json_artifact_with_dependencies, ArtifactExecutionCacheResult,
    ArtifactManifest, ContentDigest, DependencyRef,
};

const MODES_KIND: &str = "prolate_finite_dirichlet_modes";
const PROFILE_KIND: &str = "prolate_finite_dirichlet_profile";
const MODES_SEMANTICS: &str = "prolate-fd-selected-recovered-modes-v1";
const PROFILE_SEMANTICS: &str = "prolate-fd-piecewise-linear-log-profile-v1";
const NORMALIZATION: &str =
    "unit_l2_raw_then_continuous_l2_center_positive_trapezoid_zero_integral_v1";
const SPECTRUM_SEMANTICS: &str = "prolate-fd-working-precision-spectrum-v0.15.1-v2";

pub(super) struct Modes {
    pub h0_vec: Vec<Float>,
    pub h4_vec: Vec<Float>,
    pub eigenvalue_0: Float,
    pub eigenvalue_4: Float,
    pub h0_window: (Float, Float),
    pub h4_window: (Float, Float),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Mode {
    index: usize,
    vector: Vec<String>,
    eigenvalue: String,
    lower: String,
    upper: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PortableModes {
    schema_version: u32,
    identity: ContentDigest,
    modes: Vec<Mode>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PortableProfile {
    schema_version: u32,
    identity: ContentDigest,
    u_grid: Vec<String>,
    k_values: Vec<String>,
}

fn invalid(message: impl ToString) -> CacheError {
    CacheError::InvalidManifest(message.to_string())
}

fn scalar(value: &str, p: u32) -> Result<Float, CacheError> {
    let value = Float::with_val(p, Float::parse(value).map_err(invalid)?);
    if !value.is_finite() {
        return Err(invalid("nonfinite finite-Dirichlet artifact scalar"));
    }
    Ok(value)
}

fn manifest<T>(source: &ArtifactExecutionCacheResult<T>) -> Option<&ArtifactManifest> {
    source
        .produced_manifest
        .as_ref()
        .or(source.reused_manifest.as_ref())
}

fn dependencies(source: Option<&ArtifactManifest>) -> Vec<DependencyRef> {
    source
        .into_iter()
        .map(|m| DependencyRef {
            key: m.key.clone(),
            content_digest: m.content_digest.clone(),
            required_quality: CacheQuality::Validated,
        })
        .collect()
}

fn check_sources<T>(
    result: &ArtifactExecutionCacheResult<T>,
    semantic: &SemanticKeyEnvelope,
    expected: &[DependencyRef],
    cache: &ArtifactCacheContext<'_>,
) -> Result<(), CacheError> {
    if let Some(m) = manifest(result) {
        xc_cache::validate_artifact_dependency_closure(m, semantic, "prolate", expected, cache)?;
    }
    Ok(())
}

fn key(
    lambda: &Float,
    n: usize,
    p: u32,
    source: &ContentDigest,
    sample_count: Option<usize>,
) -> SemanticKeyEnvelope {
    let profile = sample_count.is_some();
    SemanticKeyEnvelope {
        schema_version: 1,
        artifact_kind: if profile { PROFILE_KIND } else { MODES_KIND }.into(),
        mathematical_semantics_version: if profile {
            PROFILE_SEMANTICS
        } else {
            MODES_SEMANTICS
        }
        .into(),
        resolved_mathematical_parameters: serde_json::json!({
            "lambda": lambda.to_string(), "lambda_precision_bits": lambda.prec(),
            "grid_points": n, "precision_bits": p, "sample_count": sample_count,
            "scalar_backend": "rug_mpfr", "selected_indices": [0, 4],
            "discretization": "centered_finite_difference_dirichlet_v1",
            "spectrum_semantics": SPECTRUM_SEMANTICS,
            "spectrum_algorithm": xc_numerics::eigen::TRIDIAG_QR_SEMANTICS,
            "recovery_semantics": xc_numerics::eigen::DENSE_EIGENVECTOR_SEMANTICS,
            "inverse_iteration_solver": "banded_interleaved", "maximum_steps": 200,
            "early_termination": true, "search_depth": super::super::PROLATE_SEARCH_DEPTH,
            "normalization": NORMALIZATION,
        }),
        normalization: Some(NORMALIZATION.into()),
        target: Some("prolate_wave_operator".into()),
        subspace: Some("even_modes_0_4".into()),
        source_data_identities: BTreeMap::from([(
            if profile {
                "selected_modes"
            } else {
                "complete_spectrum"
            }
            .into(),
            source.clone(),
        )]),
        algorithm_semantics: Some(if profile {
            "log_grid_mpfr_exp_piecewise_linear_open_support_sequential_sum_v1".into()
        } else {
            format!(
                "{};banded_interleaved;steps200;early_termination",
                xc_numerics::eigen::DENSE_EIGENVECTOR_SEMANTICS
            )
        }),
    }
}

fn request<'a>(
    semantic: &'a SemanticKeyEnvelope,
    logical: &'a str,
    cache: &'a ArtifactCacheContext<'_>,
) -> Result<ArtifactExecutionCacheRequest<'a>, CacheError> {
    Ok(ArtifactExecutionCacheRequest {
        operation: if semantic.artifact_kind == MODES_KIND {
            "prolate.finite_dirichlet_modes.resolve_or_compute"
        } else {
            "prolate.finite_dirichlet_profile.resolve_or_compute"
        },
        semantic_key: semantic,
        logical_key: logical,
        resolver: cache.resolver,
        reference_resolver: cache.reference_resolver,
        acceptance: cache.acceptance,
        ordered_overlays: cache.ordered_overlays.clone(),
        mode: cache.mode,
        write_on_miss: cache.write_on_miss,
        write_visibility: cache.write_visibility,
        produced_quality: CacheQuality::Validated,
        producer_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION"))?,
        minimum_reader_version: ToolkitVersion::parse(xc_cache::CLEAN_SLATE)?,
        maximum_reader_version: None,
        tags: BTreeMap::from([("domain".into(), "prolate".into())]),
        provenance_digest: None,
        production_sink: cache.production_sink,
    })
}

fn encode_mode(index: usize, vector: &[Float], value: &Float, window: &(Float, Float)) -> Mode {
    Mode {
        index,
        vector: vector.iter().map(Float::to_string).collect(),
        eigenvalue: value.to_string(),
        lower: window.0.to_string(),
        upper: window.1.to_string(),
    }
}

fn decode_modes(
    artifact: &PortableModes,
    identity: &ContentDigest,
    d: &[Float],
    e: &[Float],
    p: u32,
) -> Result<Modes, CacheError> {
    if artifact.schema_version != 1 || &artifact.identity != identity || artifact.modes.len() != 2 {
        return Err(invalid("finite-Dirichlet mode payload identity mismatch"));
    }
    let w = (2 * p + 64).min(1_000_000);
    let mut decoded = Vec::new();
    for (mode, index) in artifact.modes.iter().zip([0, 4]) {
        if mode.index != index || mode.vector.len() != d.len() {
            return Err(invalid("finite-Dirichlet mode shape/index mismatch"));
        }
        let vector: Vec<_> = mode
            .vector
            .iter()
            .map(|x| scalar(x, p))
            .collect::<Result<_, _>>()?;
        let value = scalar(&mode.eigenvalue, p)?;
        let lower = scalar(&mode.lower, w)?;
        let upper = scalar(&mode.upper, w)?;
        // Recovery returns unit Euclidean norm before the historical scaling.
        let norm = vector.iter().fold(Float::with_val(w, 0), |mut sum, x| {
            sum += Float::with_val(w, x).square();
            sum
        });
        if Float::with_val(w, norm - 1).abs() > (Float::with_val(w, 8 * d.len()) >> p)
            || !matches!(parity_of(&vector, p), HpParity::Even)
            || try_count_nodes(&vector, p).map_err(invalid)? != index
        {
            return Err(invalid(
                "finite-Dirichlet mode normalization/parity/nodes mismatch",
            ));
        }
        let report = xc_numerics::eigen::validate_tridiag_eigenvector_hp(
            d, e, &vector, index, &lower, &upper, p,
        )
        .map_err(invalid)?;
        if report.eigenvalue != value {
            return Err(invalid("finite-Dirichlet mode Rayleigh value mismatch"));
        }
        decoded.push((vector, value, (lower, upper)));
    }
    let (h4_vec, eigenvalue_4, h4_window) = decoded.pop().expect("two modes");
    let (h0_vec, eigenvalue_0, h0_window) = decoded.pop().expect("two modes");
    Ok(Modes {
        h0_vec,
        h4_vec,
        eigenvalue_0,
        eigenvalue_4,
        h0_window,
        h4_window,
    })
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(super) fn modes_via_cache(
    lambda: &Float,
    n: usize,
    p: u32,
    d: &[Float],
    e: &[Float],
    eigenvalues: &[Float],
    spectrum: &ArtifactExecutionCacheResult<PortableProlateSpectrum>,
    cache: &ArtifactCacheContext<'_>,
) -> Result<(Modes, Option<ArtifactManifest>, Option<ContentDigest>), CacheError> {
    let spectrum_digest = ContentDigest::sha256_finite_json(&spectrum.value)?;
    let semantic = key(lambda, n, p, &spectrum_digest, None);
    let identity = semantic.digest()?;
    let logical = format!("prolate/finite-dirichlet-modes/{}", identity.0);
    let request = request(&semantic, &logical, cache)?;
    let parents = dependencies(manifest(spectrum));
    let decoded = std::cell::RefCell::new(None);
    let resolved = resolve_or_compute_json_artifact_with_dependencies(
        &request,
        || {
            let modes = recover_finite_dirichlet_modes(d, e, eigenvalues, p).map_err(invalid)?;
            Ok((
                PortableModes {
                    schema_version: 1,
                    identity: identity.clone(),
                    modes: vec![
                        encode_mode(0, &modes.h0_vec, &modes.eigenvalue_0, &modes.h0_window),
                        encode_mode(4, &modes.h4_vec, &modes.eigenvalue_4, &modes.h4_window),
                    ],
                },
                parents.clone(),
            ))
        },
        |artifact| {
            let modes = decode_modes(artifact, &identity, d, e, p)?;
            *decoded.borrow_mut() = Some(modes);
            Ok(())
        },
    )?;
    check_sources(&resolved, &semantic, &parents, cache)?;
    let digest = ContentDigest::sha256_finite_json(&resolved.value)?;
    Ok((
        decoded
            .into_inner()
            .ok_or_else(|| invalid("mode decoder was not validated"))?,
        manifest(&resolved).cloned(),
        Some(digest),
    ))
}

fn decode_profile(
    artifact: &PortableProfile,
    identity: &ContentDigest,
    lambda: &Float,
    samples: usize,
    p: u32,
) -> Result<(Vec<Float>, Vec<Float>), CacheError> {
    if artifact.schema_version != 1
        || &artifact.identity != identity
        || artifact.u_grid.len() != samples
        || artifact.k_values.len() != samples
    {
        return Err(invalid(
            "finite-Dirichlet profile payload identity/shape mismatch",
        ));
    }
    let grid: Vec<_> = artifact
        .u_grid
        .iter()
        .map(|x| scalar(x, p))
        .collect::<Result<_, _>>()?;
    let values: Vec<_> = artifact
        .k_values
        .iter()
        .map(|x| scalar(x, p))
        .collect::<Result<_, _>>()?;
    let log_lambda = Float::with_val(p, lambda).ln();
    let lower = (-log_lambda.clone()).exp();
    let upper = log_lambda.exp();
    if grid.first() != Some(&lower)
        || grid.last() != Some(&upper)
        || grid.iter().any(|u| u <= &0)
        || grid.windows(2).any(|pair| pair[0] > pair[1])
    {
        return Err(invalid(
            "finite-Dirichlet profile grid order/endpoints mismatch",
        ));
    }
    // Rounded duplicate points remain admissible, matching the producer at
    // low precision or cutoffs close to one. Interior grid points, like samples,
    // retain computed assurance; reference verification replays every point.
    // Finite values and authenticated exact parent content establish computed
    // assurance. VerifyAgainstReference recomputes the entire sampling loop;
    // ordinary reuse does not certify pointwise accuracy or replay that loop.
    Ok((grid, values))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn profile_via_cache(
    lambda: &Float,
    n: usize,
    samples: usize,
    p: u32,
    h_lambda: &[Float],
    modes_digest: &ContentDigest,
    modes_manifest: Option<&ArtifactManifest>,
    cache: &ArtifactCacheContext<'_>,
) -> Result<(Vec<Float>, Vec<Float>), CacheError> {
    let semantic = key(lambda, n, p, modes_digest, Some(samples));
    let identity = semantic.digest()?;
    let logical = format!("prolate/finite-dirichlet-profile/{}", identity.0);
    let request = request(&semantic, &logical, cache)?;
    let parents = dependencies(modes_manifest);
    let decoded = std::cell::RefCell::new(None);
    let resolved = resolve_or_compute_json_artifact_with_dependencies(
        &request,
        || {
            let working_lambda = Float::with_val(p, lambda);
            let (grid, values) =
                sample_finite_dirichlet_profile(&working_lambda, h_lambda, samples, p)
                    .map_err(invalid)?;
            Ok((
                PortableProfile {
                    schema_version: 1,
                    identity: identity.clone(),
                    u_grid: grid.iter().map(Float::to_string).collect(),
                    k_values: values.iter().map(Float::to_string).collect(),
                },
                parents.clone(),
            ))
        },
        |artifact| {
            let profile = decode_profile(artifact, &identity, lambda, samples, p)?;
            *decoded.borrow_mut() = Some(profile);
            Ok(())
        },
    )?;
    check_sources(&resolved, &semantic, &parents, cache)?;
    decoded
        .into_inner()
        .ok_or_else(|| invalid("profile decoder was not validated"))
}

#[cfg(test)]
std::thread_local! {
    static COMPUTATIONS: std::cell::Cell<(usize, usize)> = const { std::cell::Cell::new((0, 0)) };
}
#[cfg(test)]
pub(super) fn mode_computed() {
    COMPUTATIONS.with(|c| {
        let (m, p) = c.get();
        c.set((m + 1, p));
    });
}
#[cfg(test)]
pub(super) fn profile_computed() {
    COMPUTATIONS.with(|c| {
        let (m, p) = c.get();
        c.set((m, p + 1));
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compare_shared_recovery(n: usize, p: u32) {
        let lambda = Float::with_val(p, 13).sqrt();
        let (d, e) = build_pw_matrix(&lambda, n, p);
        let values = xc_numerics::eigen::tridiag_eigenvalues_hp(&d, &e, p).unwrap();
        let start = std::time::Instant::now();
        let mut attempts = 0;
        let fresh = recover_finite_dirichlet_modes_with(n, &values, p, |value| {
            attempts += 1;
            xc_numerics::eigen::tridiag_eigenvector_for_value_detailed_hp(
                &d, &e, value, None, p, TridiagEigvecOptions::default(),
            )
        }).unwrap();
        let fresh_seconds = start.elapsed().as_secs_f64();
        let start = std::time::Instant::now();
        let shared = recover_finite_dirichlet_modes(&d, &e, &values, p).unwrap();
        let shared_seconds = start.elapsed().as_secs_f64();
        let portable = |m: &Modes| PortableModes {
            schema_version: 1,
            identity: ContentDigest::sha256(b"shared-guide-equivalence"),
            modes: vec![encode_mode(0, &m.h0_vec, &m.eigenvalue_0, &m.h0_window),
                        encode_mode(4, &m.h4_vec, &m.eigenvalue_4, &m.h4_window)],
        };
        assert_eq!(serde_json::to_vec(&portable(&fresh)).unwrap(), serde_json::to_vec(&portable(&shared)).unwrap());
        assert_eq!(attempts, 5);
        println!("shared guide comparison: grid={n}, precision={p}, attempts={attempts}, fresh_seconds={fresh_seconds:.6}, shared_seconds={shared_seconds:.6}; mode payload bytes identical");
    }

    #[test]
    fn fd_shared_guide_preserves_mode_payload_and_failed_search_behavior() {
        compare_shared_recovery(41, 64);
        compare_shared_recovery(101, 128);
        let values: Vec<_> = (0..5).map(|x| Float::with_val(64, x)).collect();
        let mut attempts = 0;
        let failed = recover_finite_dirichlet_modes_with(5, &values, 64, |_| {
            attempts += 1;
            anyhow::bail!("guide unavailable")
        });
        assert_eq!(attempts, 5);
        assert!(failed.err().unwrap().to_string().contains("could not find h_{0,λ}"));
    }

    #[test]
    #[ignore = "bounded high-precision timing probe; run explicitly"]
    fn fd_shared_guide_timing_probe() {
        compare_shared_recovery(201, 3386);
    }
    use std::sync::Mutex;
    use xc_cache::{
        ArtifactDraft, ArtifactExecutionCacheMode as ModePolicy, ArtifactProductionSink,
        CacheLayer, CachePolicy, CacheResolver, CacheStore, CacheVisibility, FilesystemCacheStore,
        ProducedArtifactRecord,
    };

    #[derive(Default)]
    struct Sink(Mutex<Vec<ProducedArtifactRecord>>);
    impl ArtifactProductionSink for Sink {
        fn record(&self, artifact: ProducedArtifactRecord) -> Result<(), CacheError> {
            self.0.lock().unwrap().push(artifact);
            Ok(())
        }
        fn record_assurance(
            &self,
            _: xc_cache::ArtifactAssuranceAttestation,
        ) -> Result<(), CacheError> {
            Ok(())
        }
        fn record_assurance_requirement(
            &self,
            _: xc_cache::ArtifactAssuranceRequirement,
        ) -> Result<(), CacheError> {
            Ok(())
        }
        fn record_evidence(&self, _: &str, bytes: &[u8]) -> Result<ContentDigest, CacheError> {
            Ok(ContentDigest::sha256(bytes))
        }
    }
    fn resolver(root: &std::path::Path) -> CacheResolver {
        CacheResolver::new(vec![CacheLayer {
            precedence: 0,
            store: Box::new(FilesystemCacheStore::new(
                "fixture",
                root,
                true,
                CacheVisibility::Local,
            )),
        }])
    }
    fn policy() -> CachePolicy {
        CachePolicy {
            current_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION")).unwrap(),
            minimum_quality: CacheQuality::Validated,
            accepted_schema_versions: vec![1],
            allow_deprecated: false,
            allow_quarantined: false,
            allowed_visibilities: vec![CacheVisibility::Local],
        }
    }
    fn context<'a>(
        resolver: &'a CacheResolver,
        policy: &'a CachePolicy,
        sink: &'a Sink,
    ) -> ArtifactCacheContext<'a> {
        ArtifactCacheContext {
            resolver: Some(resolver),
            reference_resolver: None,
            acceptance: Some(policy),
            ordered_overlays: vec!["fixture".into()],
            mode: ModePolicy::PreferReuse,
            write_on_miss: true,
            write_visibility: CacheVisibility::Local,
            requested_assurance: xc_core::AssuranceLevel::Computed,
            certification_failure_policy:
                xc_cache::CertificationFailurePolicy::RetainComputedFailRun,
            production_sink: Some(sink),
        }
    }
    fn counts() -> (usize, usize) {
        COMPUTATIONS.with(std::cell::Cell::get)
    }
    fn reset() {
        COMPUTATIONS.with(|c| c.set((0, 0)));
    }
    fn same(a: &HpProlateResult, b: &HpProlateResult) {
        assert_eq!(a.k_values, b.k_values);
        assert_eq!(a.u_grid, b.u_grid);
        assert_eq!(a.eigenvalue_0, b.eigenvalue_0);
        assert_eq!(a.eigenvalue_4, b.eigenvalue_4);
        assert_eq!(a.c_0, b.c_0);
        assert_eq!(a.c_4, b.c_4);
        assert_eq!(a.discretization, b.discretization);
        assert_eq!(a.basis_dimension, b.basis_dimension);
        assert_eq!(a.precision_bits, b.precision_bits);
    }

    #[test]
    fn fd_cache_public_route_reuses_modes_independently_of_samples_and_refreshes() {
        let root = xc_core::test_support::TestDir::new("fd-modes-profile");
        let resolver = resolver(&root);
        let policy = policy();
        let sink = Sink::default();
        let cache = context(&resolver, &policy, &sink);
        let lambda = Float::with_val(128, Float::parse("1.7").unwrap());
        reset();
        let baseline =
            compute_k_lambda_finite_dirichlet(&lambda, 33, 7, 128, CacheMode::Off).unwrap();
        assert_eq!(counts(), (1, 1));
        reset();
        let cold =
            compute_k_lambda_finite_dirichlet_via_cache(&lambda, 33, 7, 128, &cache).unwrap();
        same(&baseline, &cold);
        assert_eq!(counts(), (1, 1));
        let required = ArtifactCacheContext {
            mode: ModePolicy::RequireReuse,
            write_on_miss: false,
            ..cache
        };
        let warm =
            compute_k_lambda_finite_dirichlet_via_cache(&lambda, 33, 7, 128, &required).unwrap();
        same(&cold, &warm);
        assert_eq!(counts(), (1, 1));
        // A missing profile under require-reuse cannot fall back to sampling.
        assert!(
            compute_k_lambda_finite_dirichlet_via_cache(&lambda, 33, 9, 128, &required).is_err()
        );
        assert_eq!(counts(), (1, 1));
        let cache = context(&resolver, &policy, &sink);
        let more_samples =
            compute_k_lambda_finite_dirichlet_via_cache(&lambda, 33, 9, 128, &cache).unwrap();
        assert_eq!(more_samples.k_values.len(), 9);
        assert_eq!(counts(), (1, 2));
        let records = sink.0.lock().unwrap();
        let modes = records
            .iter()
            .find(|r| r.semantic_key.artifact_kind == MODES_KIND)
            .unwrap();
        assert_eq!(modes.manifest.dependencies.len(), 1);
        assert_eq!(
            modes.manifest.dependencies[0].key.kind,
            "prolate_eigenvalue_spectrum"
        );
        let profile = records
            .iter()
            .find(|r| r.semantic_key.artifact_kind == PROFILE_KIND)
            .unwrap();
        assert_eq!(profile.manifest.dependencies.len(), 1);
        assert_eq!(profile.manifest.dependencies[0].key, modes.manifest.key);
        assert_eq!(
            profile.manifest.dependencies[0].content_digest,
            modes.manifest.content_digest
        );
        drop(records);
        let refresh = ArtifactCacheContext {
            mode: ModePolicy::Refresh,
            ..cache
        };
        let refreshed =
            compute_k_lambda_finite_dirichlet_via_cache(&lambda, 33, 7, 128, &refresh).unwrap();
        same(&cold, &refreshed);
        assert_eq!(counts(), (2, 3));
    }

    #[test]
    fn fd_cache_identities_bind_exact_source_precision_grid_algorithm_and_spectrum() {
        let lambda = Float::with_val(160, Float::parse("1.7").unwrap());
        let source = ContentDigest::sha256(b"complete spectrum");
        let base = key(&lambda, 33, 128, &source, None);
        let mut near = lambda.clone();
        near.next_up();
        let variants = [
            key(&near, 33, 128, &source, None),
            key(&Float::with_val(161, &lambda), 33, 128, &source, None),
            key(&lambda, 35, 128, &source, None),
            key(&lambda, 33, 129, &source, None),
            key(
                &lambda,
                33,
                128,
                &ContentDigest::sha256(b"other spectrum"),
                None,
            ),
        ];
        for changed in variants {
            assert_ne!(base.digest().unwrap(), changed.digest().unwrap());
        }
        let mut algorithm = base.clone();
        algorithm.algorithm_semantics = Some("other algorithm".into());
        assert_ne!(base.digest().unwrap(), algorithm.digest().unwrap());
        assert_ne!(
            key(&lambda, 33, 128, &source, Some(7)).digest().unwrap(),
            key(&lambda, 33, 128, &source, Some(9)).digest().unwrap()
        );
        for visibility in [CacheVisibility::Public, CacheVisibility::Private] {
            let kinds: Vec<_> = xc_cache::artifact_kinds_for_family("prolate")
                .unwrap()
                .iter()
                .filter(|kind| {
                    visibility == CacheVisibility::Private
                        || !xc_cache::artifact_kind_is_private_only(kind)
                })
                .map(|k| (*k).into())
                .collect();
            assert!(xc_cache::audit_artifact_kind_registration(
                "prolate", visibility, &kinds, &kinds
            )
            .unwrap()
            .is_ready());
            assert!(kinds.iter().any(|k: &String| k == MODES_KIND));
            assert!(kinds.iter().any(|k: &String| k == PROFILE_KIND));
        }
    }

    fn copy_record(store: &FilesystemCacheStore, record: &ProducedArtifactRecord, payload: &[u8]) {
        let m = &record.manifest;
        let draft = ArtifactDraft {
            schema_version: m.schema_version,
            key: m.key.clone(),
            producer_toolkit_version: m.producer_toolkit_version.clone(),
            minimum_reader_version: m.minimum_reader_version.clone(),
            maximum_reader_version: m.maximum_reader_version.clone(),
            quality: m.quality,
            visibility: m.visibility,
            immutable: true,
            dependencies: m.dependencies.clone(),
            tags: m.tags.clone(),
            provenance_digest: m.provenance_digest.clone(),
        };
        store.put(&draft, payload).unwrap();
    }

    #[test]
    fn fd_cache_refuses_authenticated_bad_vectors_and_required_semantic_misses() {
        let root = xc_core::test_support::TestDir::new("fd-modes-poison");
        let good = resolver(&root.join("good"));
        let policy = policy();
        let sink = Sink::default();
        let cache = context(&good, &policy, &sink);
        let lambda = Float::with_val(128, Float::parse("1.7").unwrap());
        reset();
        compute_k_lambda_finite_dirichlet_via_cache(&lambda, 33, 7, 128, &cache).unwrap();
        let records = sink.0.lock().unwrap();
        let poisoned_root = root.join("poisoned");
        let store =
            FilesystemCacheStore::new("fixture", &poisoned_root, true, CacheVisibility::Local);
        for record in records.iter() {
            if record.semantic_key.artifact_kind == MODES_KIND {
                let mut artifact: PortableModes = serde_json::from_slice(&record.payload).unwrap();
                // A symmetric, finite perturbation preserves the rough shape;
                // the directed source-bound residual gate must reject it.
                let mut vector: Vec<_> = artifact.modes[0]
                    .vector
                    .iter()
                    .map(|x| scalar(x, 128).unwrap())
                    .collect();
                vector[16] += Float::with_val(128, 1) >> 30;
                let norm = vector
                    .iter()
                    .fold(Float::with_val(128, 0), |s, x| {
                        s + Float::with_val(128, x).square()
                    })
                    .sqrt();
                for x in &mut vector {
                    *x /= &norm;
                }
                artifact.modes[0].vector = vector.iter().map(Float::to_string).collect();
                copy_record(&store, record, &serde_json::to_vec(&artifact).unwrap());
            } else {
                copy_record(&store, record, &record.payload);
            }
        }
        drop(records);
        let bad = resolver(&poisoned_root);
        let bad_cache = context(&bad, &policy, &sink);
        reset();
        assert!(
            compute_k_lambda_finite_dirichlet_via_cache(&lambda, 33, 7, 128, &bad_cache).is_err()
        );
        assert_eq!(counts(), (0, 0), "poison must not silently recompute");
        let required = ArtifactCacheContext {
            mode: ModePolicy::RequireReuse,
            write_on_miss: false,
            ..cache
        };
        for (l, n, p) in [
            (Float::with_val(129, &lambda), 33, 128),
            (lambda.clone(), 35, 128),
            (lambda.clone(), 33, 129),
        ] {
            assert!(compute_k_lambda_finite_dirichlet_via_cache(&l, n, 7, p, &required).is_err());
        }
        assert_eq!(counts(), (0, 0));
    }

    #[test]
    fn fd_cache_refuses_missing_source_edges_without_recomputation() {
        let root = xc_core::test_support::TestDir::new("fd-cache-parent-refusal");
        let good = resolver(&root.join("good"));
        let policy = policy();
        let sink = Sink::default();
        let cache = context(&good, &policy, &sink);
        let lambda = Float::with_val(128, Float::parse("1.7").unwrap());
        compute_k_lambda_finite_dirichlet_via_cache(&lambda, 33, 7, 128, &cache).unwrap();
        let records = sink.0.lock().unwrap().clone();
        for kind in [MODES_KIND, PROFILE_KIND] {
            let bad_root = root.join(kind);
            let store =
                FilesystemCacheStore::new("fixture", &bad_root, true, CacheVisibility::Local);
            for record in &records {
                let mut changed = record.clone();
                if changed.semantic_key.artifact_kind == kind {
                    changed.manifest.dependencies.clear();
                }
                copy_record(&store, &changed, &changed.payload);
            }
            let bad = resolver(&bad_root);
            let cache = ArtifactCacheContext {
                mode: ModePolicy::RequireReuse,
                write_on_miss: false,
                production_sink: None,
                ..context(&bad, &policy, &sink)
            };
            reset();
            let error = compute_k_lambda_finite_dirichlet_via_cache(&lambda, 33, 7, 128, &cache)
                .unwrap_err();
            assert!(
                error.to_string().contains("dependency closure mismatch"),
                "{error}"
            );
            assert_eq!(counts(), (0, 0));
        }
    }

    #[test]
    fn fd_cache_profile_structural_gates_preserve_rounded_duplicate_points() {
        let p = 33;
        let mut lambda = Float::with_val(p, 1);
        lambda += Float::with_val(p, 1) >> 32;
        let (grid, values) =
            sample_finite_dirichlet_profile(&lambda, &vec![Float::with_val(p, 0); 17], 17, p)
                .unwrap();
        assert!(grid.windows(2).any(|pair| pair[0] == pair[1]));
        let identity = ContentDigest::sha256(b"profile structural fixture");
        let artifact = PortableProfile {
            schema_version: 1,
            identity: identity.clone(),
            u_grid: grid.iter().map(Float::to_string).collect(),
            k_values: values.iter().map(Float::to_string).collect(),
        };
        decode_profile(&artifact, &identity, &lambda, 17, p).unwrap();
        let mut changed = artifact.clone();
        changed.u_grid[0] = "0".into();
        assert!(decode_profile(&changed, &identity, &lambda, 17, p).is_err());
        let mut changed = artifact.clone();
        changed.k_values[3] = "NaN".into();
        assert!(decode_profile(&changed, &identity, &lambda, 17, p).is_err());
        let mut changed = artifact.clone();
        changed.identity = ContentDigest::sha256(b"wrong parent");
        assert!(decode_profile(&changed, &identity, &lambda, 17, p).is_err());
        assert!(decode_profile(&artifact, &identity, &lambda, 19, p).is_err());
    }

    #[test]
    fn fd_cache_stages_exact_three_layer_closure_for_public_and_private_inventory() {
        use xc_cache::{
            ManagedArtifactCacheConfig, ManagedArtifactCacheSession, ManagedRemoteCacheMode,
            ManagedRunProfile,
        };
        let root = xc_core::test_support::TestDir::new("fd-cache-publication");
        let session = ManagedArtifactCacheSession::with_layers_for_test(
            ManagedArtifactCacheConfig {
                profile: ManagedRunProfile::Author,
                requested_assurance: xc_core::AssuranceLevel::Computed,
                certification_failure_policy:
                    xc_cache::CertificationFailurePolicy::RetainComputedFailRun,
                cache_root: root.join("cache"),
                staging_root: Some(root.join("staging")),
                publication_target: xc_core::PublicationTarget::Both,
                repository_owner: "local-fixture".into(),
                remote_cache_mode: ManagedRemoteCacheMode::None,
                cache_mode: ModePolicy::PreferReuse,
                replace_existing_publication: false,
                execute_remote_mutations: false,
                output_validation: None,
            },
            vec![CacheLayer {
                precedence: 0,
                store: Box::new(FilesystemCacheStore::new(
                    "workstation",
                    root.join("cache"),
                    true,
                    CacheVisibility::Local,
                )),
            }],
            vec![],
        )
        .unwrap();
        let lambda = Float::with_val(128, Float::parse("1.7").unwrap());
        compute_k_lambda_finite_dirichlet_via_cache(&lambda, 33, 7, 128, &session.context())
            .unwrap();
        let drafts = session.staged_drafts().unwrap();
        assert_eq!(drafts.len(), 3);
        for (kind, parent_kind) in [
            (MODES_KIND, "prolate_eigenvalue_spectrum"),
            (PROFILE_KIND, MODES_KIND),
        ] {
            let draft = drafts
                .iter()
                .find(|d| d.manifest.semantic_key.artifact_kind == kind)
                .unwrap();
            let parent = drafts
                .iter()
                .find(|d| d.manifest.semantic_key.artifact_kind == parent_kind)
                .unwrap();
            let edges = &draft.manifest.canonical_payload.dependencies;
            assert_eq!(edges.len(), 1);
            assert_eq!(edges[0].semantic_digest, parent.manifest.semantic_digest);
            assert_eq!(edges[0].manifest_digest, parent.manifest.digest().unwrap());
            assert_eq!(edges[0].payload_digest, parent.manifest.payload_digest);
        }
        let inventory = session.publication_inventory().unwrap();
        assert_eq!(inventory.entries.len(), 6);
        assert_eq!(
            inventory
                .entries
                .iter()
                .filter(|e| e.destination == xc_cache::PublicationDestination::Public)
                .count(),
            3
        );
        assert_eq!(
            inventory
                .entries
                .iter()
                .filter(|e| e.destination == xc_cache::PublicationDestination::Private)
                .count(),
            3
        );
        assert!(!inventory.remote_mutation_enabled);
        session.finalize_publication_inventory().unwrap();
    }

    fn verification_session(
        root: &std::path::Path,
        reference_root: &std::path::Path,
    ) -> xc_cache::ManagedArtifactCacheSession {
        use xc_cache::{
            ManagedArtifactCacheConfig, ManagedArtifactCacheSession, ManagedRemoteCacheMode,
            ManagedRunProfile, OutputValidationConfig,
        };
        let validation = root.join("validation");
        ManagedArtifactCacheSession::with_layers_for_test(
            ManagedArtifactCacheConfig {
                profile: ManagedRunProfile::Normal,
                requested_assurance: xc_core::AssuranceLevel::Computed,
                certification_failure_policy:
                    xc_cache::CertificationFailurePolicy::RetainComputedFailRun,
                cache_root: root.join("unused-production"),
                staging_root: None,
                publication_target: xc_core::PublicationTarget::None,
                repository_owner: "local-fixture".into(),
                remote_cache_mode: ManagedRemoteCacheMode::None,
                cache_mode: ModePolicy::VerifyAgainstReference,
                replace_existing_publication: false,
                execute_remote_mutations: false,
                output_validation: Some(OutputValidationConfig {
                    validation_root: validation.clone(),
                    report_root: validation.join("reports"),
                    reference_mode: ManagedRemoteCacheMode::Public,
                }),
            },
            vec![CacheLayer {
                precedence: 0,
                store: Box::new(FilesystemCacheStore::new(
                    "validation-computed",
                    validation.join("computed"),
                    true,
                    CacheVisibility::Local,
                )),
            }],
            vec![CacheLayer {
                precedence: 0,
                store: Box::new(FilesystemCacheStore::new(
                    "reference",
                    reference_root,
                    false,
                    CacheVisibility::Local,
                )),
            }],
        )
        .unwrap()
    }

    #[test]
    fn fd_cache_verification_recomputes_all_layers_and_detects_plausible_profile_change() {
        let _lock = crate::TEST_CWD_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let root = xc_core::test_support::TestDir::new("fd-profile-verification");
        let reference_root = root.join("reference");
        let resolver = resolver(&reference_root);
        let policy = policy();
        let sink = Sink::default();
        let cache = context(&resolver, &policy, &sink);
        let lambda = Float::with_val(128, Float::parse("1.7").unwrap());
        let original =
            compute_k_lambda_finite_dirichlet_via_cache(&lambda, 33, 7, 128, &cache).unwrap();
        reset();
        let matching = verification_session(&root.join("matching"), &reference_root);
        let recomputed =
            compute_k_lambda_finite_dirichlet_via_cache(&lambda, 33, 7, 128, &matching.context())
                .unwrap();
        same(&original, &recomputed);
        assert_eq!(counts(), (1, 1));
        matching.finalize_publication_inventory().unwrap();
        let report: xc_cache::OutputPreservationValidationReport = serde_json::from_slice(
            &std::fs::read(root.join("matching/validation/reports/latest.json")).unwrap(),
        )
        .unwrap();
        assert!(report.output_preserving);
        assert_eq!(report.totals.matched, 3);
        drop(matching);
        let poisoned_root = root.join("plausible-profile");
        let store =
            FilesystemCacheStore::new("fixture", &poisoned_root, true, CacheVisibility::Local);
        for record in sink.0.lock().unwrap().iter() {
            if record.semantic_key.artifact_kind == PROFILE_KIND {
                let mut artifact: PortableProfile =
                    serde_json::from_slice(&record.payload).unwrap();
                let changed =
                    scalar(&artifact.k_values[3], 128).unwrap() + Float::with_val(128, 1) / 8u32;
                artifact.k_values[3] = changed.to_string();
                copy_record(&store, record, &serde_json::to_vec(&artifact).unwrap());
            } else {
                copy_record(&store, record, &record.payload);
            }
        }
        reset();
        let differing = verification_session(&root.join("differing"), &poisoned_root);
        let recomputed =
            compute_k_lambda_finite_dirichlet_via_cache(&lambda, 33, 7, 128, &differing.context())
                .unwrap();
        same(&original, &recomputed);
        assert_eq!(counts(), (1, 1));
        assert!(differing.finalize_publication_inventory().is_err());
        let report: xc_cache::OutputPreservationValidationReport = serde_json::from_slice(
            &std::fs::read(root.join("differing/validation/reports/latest.json")).unwrap(),
        )
        .unwrap();
        assert!(!report.output_preserving);
        assert_eq!(report.totals.matched, 2);
        assert_eq!(report.totals.mismatched, 1);
        assert_eq!(
            report
                .comparisons
                .iter()
                .find(|c| c.status == xc_cache::ArtifactOutputComparisonStatus::Mismatch)
                .unwrap()
                .semantic_key
                .artifact_kind,
            PROFILE_KIND
        );
    }
}
