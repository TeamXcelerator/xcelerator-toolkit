// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Gauss-Legendre quadrature at f64 and high precision.
//!
//! The HP nodes/weights are computed via Newton iteration on Legendre
//! polynomials and cached to disk under `$XC_CACHE_ROOT/gl_cache/`, or
//! `<cwd>/data/gl_cache/` when that variable is unset, so they're reused
//! across runs at the same `(n_pts, precision_bits)`.
//!
//! The standalone HP API reads a local deterministic `.json.zip`, directly
//! in memory, and computes and stores that representation on a miss.
//! Remote local/private/public resolution belongs exclusively to the managed
//! cache fabric exposed by `gauss_legendre_nodes_via_cache`.
//!
//! `CacheMode` selects how far down this list a lookup goes:
//! - `Off`          — no cache read or write; always compute.
//! - `JsonOnly`     — step (1) only.
//! - `JsonZip`      — local compressed cache (**default**).
//!
//! `gauss_legendre_nodes(n, prec, mode)` takes the `CacheMode`
//! explicitly; pass `CacheMode::default()` for standard local behavior.
//!
//! New computes write only the compressed representation.

/// Machine-readable quadrature artifact decomposition. Nodes and weights are
/// independent of downstream integrands and can be reused wherever order,
/// backend, precision, and generation semantics match.
pub fn quadrature_artifact_reuse_plan() -> xc_core::ArtifactReusePlan {
    use xc_core::{ArtifactReuseNode, ArtifactReusePlan};
    ArtifactReusePlan {
        schema_version: 1,
        domain: "quadrature".to_owned(),
        semantics_version: "gauss-legendre-v0.13.0-v1".to_owned(),
        artifacts: vec![
            ArtifactReuseNode {
                kind: "rule_nodes_weights".to_owned(),
                independently_cacheable: true,
                dependencies: Vec::new(),
                invalidated_by: vec![
                    "rule_family".to_owned(),
                    "order".to_owned(),
                    "scalar_backend".to_owned(),
                    "precision_bits".to_owned(),
                    "generation_semantics".to_owned(),
                ],
            },
            ArtifactReuseNode {
                kind: "rule_validation".to_owned(),
                independently_cacheable: true,
                dependencies: vec!["rule_nodes_weights".to_owned()],
                invalidated_by: vec!["validation_policy".to_owned()],
            },
        ],
    }
}

/// 64-point Gauss-Legendre quadrature on [a, b] at f64 precision.
/// For a configurable number of points, use `gauss_legendre_npt_f64`.
pub fn gauss_legendre_64pt_f64<F: Fn(f64) -> f64>(f: F, a: f64, b: f64) -> f64 {
    gauss_legendre_npt_f64(f, a, b, 64)
}

/// Approximates an integral with an `n`-point Gauss--Legendre rule.
///
/// # Mathematical semantics
/// Maps the canonical nodes on `[-1, 1]` to `[a, b]` and returns the weighted
/// sum. It is exact for polynomials through degree `2n - 1` in exact
/// arithmetic; it does not prove an error bound for a general integrand.
///
/// # Precision
/// Nodes, weights, function values, and accumulation all use binary64. Choose
/// an HP quadrature route when binary64 rounding is not an explicit policy.
///
/// # Failure states
/// This convenience function has no typed error channel. Non-finite bounds or
/// integrand values propagate through binary64 arithmetic, and callers must
/// validate them when such values are inadmissible.
///
/// # Assurance and validity
/// The result is exploratory unless independently cross-checked or enclosed by
/// a separate certified integration method.
///
/// # Cache effects
/// This function performs no cache lookup, persistence, or publication.
///
/// # Example
/// Compiled example: `crates/xc-numerics/examples/quadrature.rs`.
pub fn gauss_legendre_npt_f64<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n: usize) -> f64 {
    let (nodes, weights) = gl_nodes_weights_f64(n);
    let mid = 0.5 * (a + b);
    let half = 0.5 * (b - a);
    let mut sum = 0.0_f64;
    for i in 0..n {
        let x = mid + half * nodes[i];
        sum += weights[i] * f(x);
    }
    sum * half
}

/// Compute n-point Gauss-Legendre nodes and weights on `[-1, 1]` at f64.
///
/// The nodes are sorted in ascending order. Callers that need to set up
/// a variable-integrand integral (e.g. a complex-valued integrand with
/// multiple accumulator sums) can call this directly rather than using
/// [`gauss_legendre_npt_f64`], which takes a single closure.
pub fn gl_nodes_weights_f64(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = vec![0.0_f64; n];
    let mut weights = vec![0.0_f64; n];
    for k in 0..n {
        let mut x = ((4 * k + 3) as f64 * std::f64::consts::PI / (4 * n + 2) as f64).cos();
        for _ in 0..20 {
            let (pn, pn_prime) = legendre_p_deriv_f64(n, x);
            x -= pn / pn_prime;
        }
        let (_, pn_prime) = legendre_p_deriv_f64(n, x);
        nodes[k] = x;
        weights[k] = 2.0 / ((1.0 - x * x) * pn_prime * pn_prime);
    }
    (nodes, weights)
}

fn legendre_p_deriv_f64(n: usize, x: f64) -> (f64, f64) {
    if n == 0 {
        return (1.0, 0.0);
    }
    let mut p0 = 1.0_f64;
    let mut p1 = x;
    for k in 1..n {
        let p_next = ((2 * k + 1) as f64 * x * p1 - k as f64 * p0) / (k + 1) as f64;
        p0 = p1;
        p1 = p_next;
    }
    let deriv = n as f64 * (x * p1 - p0) / (x * x - 1.0);
    (p1, deriv)
}

// ===========================================================================
// High-precision Gauss-Legendre nodes and weights (with disk cache)
// ===========================================================================

#[cfg(feature = "hp")]
mod hp {
    use rug::{ops::Pow, Float};
    use serde::{Deserialize, Serialize};
    use std::collections::BTreeMap;
    use xc_cache::{
        resolve_or_compute_json_artifact, ArtifactCacheContext, ArtifactExecutionCacheRequest,
        CacheError, CacheQuality, SemanticKeyEnvelope, ToolkitVersion,
    };
    use xc_core::CacheAccessProvenance;

    /// A Gauss-Legendre quadrature table: `(nodes, weights)`, each vector
    /// of length `n` on `[-1, 1]`.
    type GlTable = (Vec<Float>, Vec<Float>);

    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct PortableGlTable {
        schema_version: u32,
        order: usize,
        precision_bits: u32,
        nodes: Vec<String>,
        weights: Vec<String>,
    }

    /// Cache-fabric controls for one HP Gauss--Legendre rule resolution.
    pub type QuadratureCacheRequest<'a> = ArtifactCacheContext<'a>;

    /// An HP Gauss--Legendre rule and its auditable cache decision.
    pub struct CachedQuadratureRule {
        pub nodes: Vec<Float>,
        pub weights: Vec<Float>,
        pub cache_access: CacheAccessProvenance,
        pub artifact_manifest: xc_cache::ArtifactManifest,
    }

    fn portable_table(n: usize, prec: u32, table: GlTable) -> PortableGlTable {
        PortableGlTable {
            schema_version: 1,
            order: n,
            precision_bits: prec,
            nodes: table.0.iter().map(Float::to_string).collect(),
            weights: table.1.iter().map(Float::to_string).collect(),
        }
    }

    fn decode_portable_table(
        table: &PortableGlTable,
        n: usize,
        prec: u32,
    ) -> Result<GlTable, CacheError> {
        if table.schema_version != 1
            || table.order != n
            || table.precision_bits != prec
            || table.nodes.len() != n
            || table.weights.len() != n
        {
            return Err(CacheError::InvalidManifest(
                "quadrature payload identity or dimensions do not match its semantic key"
                    .to_owned(),
            ));
        }
        let parse = |value: &str| {
            Float::parse(value)
                .map(|parsed| Float::with_val(prec, parsed))
                .map_err(|error| {
                    CacheError::InvalidManifest(format!(
                        "quadrature payload contains an invalid HP scalar: {error}"
                    ))
                })
        };
        let nodes: Vec<Float> = table
            .nodes
            .iter()
            .map(|value| parse(value))
            .collect::<Result<_, _>>()?;
        let weights: Vec<Float> = table
            .weights
            .iter()
            .map(|value| parse(value))
            .collect::<Result<_, _>>()?;
        if let Some(reason) = cache_structural_check(&nodes, &weights, prec) {
            return Err(CacheError::InvalidManifest(format!(
                "quadrature payload failed structural validation: {reason}"
            )));
        }
        Ok((nodes, weights))
    }

    /// Resolves or computes an HP Gauss--Legendre rule through the common cache fabric.
    ///
    /// # Mathematical semantics
    /// The artifact is the ordered Gauss--Legendre nodes and weights on `[-1, 1]`.
    /// Its semantic identity includes the order, MPFR precision, normalization, and
    /// v0.13.0 generation semantics.
    ///
    /// # Precision
    /// Both computation and decoding round at `precision_bits` using `rug::Float`.
    /// Decimal payload strings preserve substantially more than binary64 precision.
    ///
    /// # Failure states
    /// Invalid inputs, cache corruption, incompatible policy, missing required reuse,
    /// and unavailable writable overlays return a typed [`CacheError`]. Corruption
    /// never silently falls through to a fresh computation.
    ///
    /// # Assurance and validity
    /// Reused and fresh rules must pass weight-sum, first-moment, and mirror-identity
    /// checks at the requested precision before they are returned.
    ///
    /// # Cache effects
    /// Behavior is entirely controlled by `cache`. `PreferReuse` may write only when
    /// `write_on_miss` is true; `RequireReuse` never computes or writes; `Disabled`
    /// performs no cache I/O. Remote access is delegated to configured overlays.
    ///
    /// # Example
    /// See the cache-fabric round-trip test in this module for a local overlay setup.
    pub fn gauss_legendre_nodes_via_cache(
        n: usize,
        precision_bits: u32,
        cache: QuadratureCacheRequest<'_>,
    ) -> Result<CachedQuadratureRule, CacheError> {
        gauss_legendre_nodes_via_cache_impl(
            n,
            precision_bits,
            cache,
            crate::hp_runtime::GlRootSchedule::serial(),
            true,
        )
    }

    /// Schedule-aware managed resolver used by an owning HP batch planner.
    /// Existing public GL entry points remain root-serial compatibility
    /// wrappers; callers must not infer scheduling inside this function.
    #[doc(hidden)]
    pub fn gauss_legendre_nodes_via_cache_scheduled(
        n: usize,
        precision_bits: u32,
        cache: QuadratureCacheRequest<'_>,
        root_schedule: crate::hp_runtime::GlRootSchedule,
    ) -> Result<CachedQuadratureRule, CacheError> {
        gauss_legendre_nodes_via_cache_impl(n, precision_bits, cache, root_schedule, false)
    }

    fn gauss_legendre_nodes_via_cache_impl(
        n: usize,
        precision_bits: u32,
        cache: QuadratureCacheRequest<'_>,
        root_schedule: crate::hp_runtime::GlRootSchedule,
        top_level: bool,
    ) -> Result<CachedQuadratureRule, CacheError> {
        let mut performance_resolve = if top_level {
            xc_core::performance_top_level_stage_with("quadrature.gl.resolve", || {
                gl_performance_metadata_scheduled(n, precision_bits, root_schedule)
            })
        } else {
            xc_core::performance_stage_with("quadrature.gl.resolve", || {
                gl_performance_metadata_scheduled(n, precision_bits, root_schedule)
            })
        };
        if n == 0 || precision_bits < 16 {
            return Err(CacheError::InvalidManifest(
                "quadrature order must be positive and precision must be at least 16 bits"
                    .to_owned(),
            ));
        }
        let semantic_key = SemanticKeyEnvelope {
            schema_version: 1,
            artifact_kind: "gauss_legendre_rule".to_owned(),
            mathematical_semantics_version: "gauss-legendre-v0.13.0-v1".to_owned(),
            resolved_mathematical_parameters: serde_json::json!({
                "order": n,
                "precision_bits": precision_bits,
                "scalar_backend": "rug_mpfr",
                "generation_semantics": "newton-legendre-roots-v1"
            }),
            normalization: Some("nodes_on_minus_one_one_weights_sum_two".to_owned()),
            target: None,
            subspace: None,
            source_data_identities: BTreeMap::new(),
            algorithm_semantics: None,
        };
        let logical_key = format!("gauss-legendre/{n}/{precision_bits}");
        let execution_request = ArtifactExecutionCacheRequest {
            operation: "quadrature.gauss_legendre.resolve_or_compute",
            semantic_key: &semantic_key,
            logical_key: &logical_key,
            resolver: cache.resolver,
            reference_resolver: cache.reference_resolver,
            acceptance: cache.acceptance,
            ordered_overlays: cache.ordered_overlays,
            mode: cache.mode,
            write_on_miss: cache.write_on_miss,
            write_visibility: cache.write_visibility,
            produced_quality: CacheQuality::Validated,
            producer_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION"))?,
            minimum_reader_version: ToolkitVersion::parse("0.13.0")?,
            maximum_reader_version: None,
            tags: BTreeMap::from([
                ("domain".to_owned(), "quadrature".to_owned()),
                ("rule_family".to_owned(), "gauss_legendre".to_owned()),
            ]),
            provenance_digest: None,
            production_sink: cache.production_sink,
        };
        let resolved = resolve_or_compute_json_artifact(
            &execution_request,
            || {
                let performance_construct =
                    xc_core::performance_stage_with("quadrature.gl.construct", || {
                        gl_performance_metadata_scheduled(n, precision_bits, root_schedule)
                    });
                let table = gauss_legendre_compute_scheduled(n, precision_bits, root_schedule);
                drop(performance_construct);
                let performance_encode =
                    xc_core::performance_stage_with("quadrature.gl.portable_encode", || {
                        gl_performance_metadata(n, precision_bits)
                    });
                let portable = portable_table(n, precision_bits, table);
                drop(performance_encode);
                Ok(portable)
            },
            |table| {
                let _performance_decode =
                    xc_core::performance_stage_with("quadrature.gl.validation_decode", || {
                        gl_performance_metadata(n, precision_bits)
                    });
                decode_portable_table(table, n, precision_bits).map(|_| ())
            },
        )?;
        performance_resolve.set_cache_disposition(match resolved.access.reuse_disposition {
            xc_core::CacheReuseDisposition::Recomputed => "computed",
            xc_core::CacheReuseDisposition::Reused => "reused",
            xc_core::CacheReuseDisposition::InspectedOnly => "verified",
        });
        let performance_decode =
            xc_core::performance_stage_with("quadrature.gl.return_decode", || {
                gl_performance_metadata(n, precision_bits)
            });
        let (nodes, weights) = decode_portable_table(&resolved.value, n, precision_bits)?;
        drop(performance_decode);
        let artifact_manifest = resolved
            .produced_manifest
            .or(resolved.reused_manifest)
            .ok_or_else(|| {
                CacheError::InvalidManifest(
                    "enabled quadrature cache execution returned no artifact manifest".to_owned(),
                )
            })?;
        Ok(CachedQuadratureRule {
            nodes,
            weights,
            cache_access: resolved.access,
            artifact_manifest,
        })
    }

    /// Controls how `gauss_legendre_nodes` resolves a `(n, prec)` table.
    ///
    /// Variants are ordered by how many lookup tiers they enable before
    /// falling back to a fresh Newton compute:
    ///
    /// - `Off`          — no cache at all. Always compute; never read or
    ///   write any cache file.
    /// - `JsonOnly`     — deprecated compatibility variant; computes without
    ///   reading or writing files under the zip-only cache contract.
    /// - `JsonZip`      — local `.json.zip`, read in memory, then compute.
    ///   This is the default for the standalone API. Managed remote resolution
    ///   is provided by `gauss_legendre_nodes_via_cache`.
    ///
    /// On a fresh compute, only `JsonZip` writes a `.json.zip`. `Off` and
    /// the deprecated `JsonOnly` variant write nothing.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub enum CacheMode {
        /// No caching: always compute, never touch disk or network.
        Off,
        /// Deprecated compatibility variant: no reads or writes.
        JsonOnly,
        /// Local `.json.zip`, followed by computation on a miss (default).
        #[default]
        JsonZip,
    }

    /// Toolkit version string embedded in every GL cache file written by
    /// this build. Matches `[workspace.package].version` in `Cargo.toml`.
    const TOOLKIT_VERSION: &str = env!("CARGO_PKG_VERSION");

    #[cfg(test)]
    pub fn toolkit_version_for_test() -> &'static str {
        TOOLKIT_VERSION
    }

    /// Minimum toolkit version required to use a GL cache file. Files
    /// produced by an older toolkit are treated as cache misses and
    /// recomputed. Update this constant when a change to the GL
    /// computation changes the stored values.
    fn effective_min_version() -> String {
        xc_cache::artifact_compatibility_policy("quadrature", "gauss_legendre_rule")
            .expect("quadrature compatibility policy")
            .minimum_producer_version
            .to_string()
    }

    #[inline]
    fn fl_i(prec: u32, v: i64) -> Float {
        Float::with_val(prec, v)
    }
    #[inline]
    fn pi(prec: u32) -> Float {
        Float::with_val(prec, rug::float::Constant::Pi)
    }

    /// Tolerance for structural identity checks on a loaded cache file.
    /// At precision `prec` bits, this is `2^-(prec - 8)` — 8 guard bits
    /// below working precision, which gives ~10⁻⁷⁵ at HP-256 and
    /// ~10⁻³⁰⁰ at HP-1000. Far above any plausible correct-cache rounding
    /// drift, well below any value corruption that would meaningfully
    /// affect downstream integration.
    fn cache_structural_tol(prec: u32) -> Float {
        Float::with_val(prec, 2).pow(-((prec as i32) - 8))
    }

    /// Verify the three classical structural identities of a Gauss-
    /// Legendre node/weight pair on `[-1, 1]`:
    ///
    ///   1. Σ w_i = 2 (length of [-1, 1])
    ///   2. Σ x_i · w_i = 0 (first moment of an even weight function)
    ///   3. Antisymmetry: nodes[i] + nodes[n-1-i] = 0,
    ///      weights[i] - weights[n-1-i] = 0
    ///
    /// Returns `None` if all three identities hold within
    /// `cache_structural_tol(prec)`. Returns `Some(reason)` describing
    /// the first identity that fails, with both magnitudes in the
    /// reason string for diagnostic purposes.
    ///
    /// Used by `load_gl_cache` to discard structurally-broken cache
    /// files (e.g. wrong precision, value corruption, accidental edit)
    /// before they pollute downstream HP integration.
    fn cache_structural_check(nodes: &[Float], weights: &[Float], prec: u32) -> Option<String> {
        let n = nodes.len();
        if weights.len() != n {
            return Some(format!(
                "weight count {} != node count {}",
                weights.len(),
                n
            ));
        }
        let tol = cache_structural_tol(prec);

        // Identity 1: Σ w_i = 2.
        let mut wsum = Float::with_val(prec, 0);
        for w in weights {
            wsum += w;
        }
        let mut wdiff = wsum.clone();
        wdiff -= 2u32;
        let abs_wdiff = wdiff.abs();
        if !abs_wdiff.cmp_abs(&tol).map(|o| o.is_lt()).unwrap_or(false) {
            return Some(format!(
                "Σ weights deviates from 2 by {} (tol {})",
                abs_wdiff, tol
            ));
        }

        // Identity 2: Σ x_i · w_i = 0.
        let mut moment = Float::with_val(prec, 0);
        for (x, w) in nodes.iter().zip(weights.iter()) {
            let mut t = x.clone();
            t *= w;
            moment += &t;
        }
        let abs_moment = moment.abs();
        if !abs_moment.cmp_abs(&tol).map(|o| o.is_lt()).unwrap_or(false) {
            return Some(format!(
                "first moment Σ x_i w_i deviates from 0 by {} (tol {})",
                abs_moment, tol
            ));
        }

        // Identity 3: antisymmetry of nodes; mirror symmetry of weights.
        for i in 0..(n / 2) {
            let mut node_sum = nodes[i].clone();
            node_sum += &nodes[n - 1 - i];
            let abs_ns = node_sum.abs();
            if !abs_ns.cmp_abs(&tol).map(|o| o.is_lt()).unwrap_or(false) {
                return Some(format!(
                    "antisymmetry: nodes[{}] + nodes[{}] = {} (tol {})",
                    i,
                    n - 1 - i,
                    abs_ns,
                    tol
                ));
            }
            let mut wm = weights[i].clone();
            wm -= &weights[n - 1 - i];
            let abs_wm = wm.abs();
            if !abs_wm.cmp_abs(&tol).map(|o| o.is_lt()).unwrap_or(false) {
                return Some(format!(
                    "weight mirror: weights[{}] - weights[{}] = {} (tol {})",
                    i,
                    n - 1 - i,
                    abs_wm,
                    tol
                ));
            }
        }

        None
    }

    /// Compute (or load from cache) Gauss-Legendre nodes and weights
    /// at the given precision in bits, for `n` points on `[-1, 1]`.
    ///
    /// `mode` selects the cache strategy; see `CacheMode` and the
    /// module docs for the lookup order. Pass `CacheMode::default()`
    /// for the standard local behavior. Use `gauss_legendre_nodes_via_cache`
    /// when managed local/private/public resolution is required.
    pub fn gauss_legendre_nodes(n: usize, prec: u32, mode: CacheMode) -> (Vec<Float>, Vec<Float>) {
        gauss_legendre_nodes_impl(
            n,
            prec,
            mode,
            crate::hp_runtime::GlRootSchedule::serial(),
            true,
        )
    }

    /// Schedule-aware compatibility-cache entry point used by the CCM batch
    /// planner. The scheduling decision has already been made by the owner.
    #[doc(hidden)]
    pub fn gauss_legendre_nodes_scheduled(
        n: usize,
        prec: u32,
        mode: CacheMode,
        root_schedule: crate::hp_runtime::GlRootSchedule,
    ) -> (Vec<Float>, Vec<Float>) {
        gauss_legendre_nodes_impl(n, prec, mode, root_schedule, false)
    }

    fn gauss_legendre_nodes_impl(
        n: usize,
        prec: u32,
        mode: CacheMode,
        root_schedule: crate::hp_runtime::GlRootSchedule,
        top_level: bool,
    ) -> (Vec<Float>, Vec<Float>) {
        let mut performance_resolve = if top_level {
            xc_core::performance_top_level_stage_with("quadrature.gl.resolve_compatibility", || {
                gl_performance_metadata_scheduled(n, prec, root_schedule)
            })
        } else {
            xc_core::performance_stage_with("quadrature.gl.resolve_compatibility", || {
                gl_performance_metadata_scheduled(n, prec, root_schedule)
            })
        };
        if mode != CacheMode::Off {
            if let Some(cached) = load_gl_cache(n, prec, mode) {
                performance_resolve.set_cache_disposition("reused");
                return cached;
            }
        }
        performance_resolve.set_cache_disposition("computed");
        let performance_construct =
            xc_core::performance_stage_with("quadrature.gl.construct", || {
                gl_performance_metadata_scheduled(n, prec, root_schedule)
            });
        let result = gauss_legendre_compute_scheduled(n, prec, root_schedule);
        drop(performance_construct);
        if mode != CacheMode::Off {
            let performance_store =
                xc_core::performance_stage_with("quadrature.gl.compatibility_store", || {
                    gl_performance_metadata(n, prec)
                });
            save_gl_cache(n, prec, &result.0, &result.1, mode);
            drop(performance_store);
        }
        result
    }

    fn gl_performance_metadata(n: usize, precision_bits: u32) -> xc_core::PerformanceStageMetadata {
        gl_performance_metadata_scheduled(
            n,
            precision_bits,
            crate::hp_runtime::GlRootSchedule::serial(),
        )
    }

    fn gl_performance_metadata_scheduled(
        n: usize,
        precision_bits: u32,
        root_schedule: crate::hp_runtime::GlRootSchedule,
    ) -> xc_core::PerformanceStageMetadata {
        xc_core::PerformanceStageMetadata {
            operation: Some("quadrature.gauss_legendre".to_owned()),
            requested_table_order: Some(n),
            precision_bits: Some(precision_bits),
            rayon_workers: Some(rayon::current_num_threads()),
            scheduling: Some(root_schedule.label().to_owned()),
            ..xc_core::PerformanceStageMetadata::default()
        }
    }

    #[cfg(test)]
    thread_local! {
        static TEST_CACHE_ROOT: std::cell::RefCell<Option<std::path::PathBuf>> =
            const { std::cell::RefCell::new(None) };
    }

    /// Test-only, thread-local cache placement. Never mutates the process
    /// environment or cwd, and never redirects another concurrently running
    /// test. Production continues to honor XC_CACHE_ROOT unchanged.
    #[cfg(test)]
    pub(super) fn replace_test_cache_root(
        root: Option<std::path::PathBuf>,
    ) -> Option<std::path::PathBuf> {
        TEST_CACHE_ROOT.with(|current| current.replace(root))
    }

    /// Cache directory: `$XC_CACHE_ROOT/gl_cache` when that is set, else
    /// `<cwd>/data/gl_cache`. Created on demand so fresh checkouts work
    /// without manual setup.
    ///
    /// Honouring `XC_CACHE_ROOT` matters because the fallback is relative to
    /// the *working directory*: a run started from inside a checkout drops
    /// binary cache files into that checkout, where they are neither the
    /// operator's chosen cache volume nor necessarily ignored by git.
    fn gl_cache_dir() -> Option<std::path::PathBuf> {
        #[cfg(test)]
        if let Some(root) = TEST_CACHE_ROOT.with(|current| current.borrow().clone()) {
            let dir = root.join("gl_cache");
            std::fs::create_dir_all(&dir).ok()?;
            return Some(dir);
        }
        let root = match std::env::var_os("XC_CACHE_ROOT") {
            Some(root) if !root.is_empty() => std::path::PathBuf::from(root),
            _ => std::env::current_dir().ok()?.join("data"),
        };
        let dir = root.join("gl_cache");
        std::fs::create_dir_all(&dir).ok()?;
        Some(dir)
    }

    /// Path to the uncompressed cache file for `(n, prec)`.
    fn gl_cache_path(n: usize, prec: u32) -> Option<std::path::PathBuf> {
        gl_cache_dir().map(|d| d.join(format!("prec{}_npts{}.json", prec, n)))
    }

    /// Path to the zip-compressed cache file for `(n, prec)`. The zip
    /// archive is expected to contain a single entry whose name is the
    /// uncompressed JSON filename (no `.zip` suffix).
    fn gl_cache_zip_path(n: usize, prec: u32) -> Option<std::path::PathBuf> {
        gl_cache_dir().map(|d| d.join(format!("prec{}_npts{}.json.zip", prec, n)))
    }

    /// Parse the GL cache JSON into HP node and weight vectors.
    /// Expects schema_version 1 envelope format. Returns `None` on any
    /// structural mismatch or a stale `toolkit_version`.
    fn parse_gl_json(data: &str, n: usize, prec: u32) -> Option<(Vec<Float>, Vec<Float>)> {
        let parsed: serde_json::Value = serde_json::from_str(data).ok()?;
        let obj = parsed.as_object()?;
        if obj.get("schema_version")?.as_u64()? != 1
            || obj.get("n_pts")?.as_u64()? != u64::try_from(n).ok()?
            || obj.get("precision_bits")?.as_u64()? != u64::from(prec)
        {
            return None;
        }

        let file_ver = obj.get("toolkit_version").and_then(|v| v.as_str())?;
        if version_is_older(file_ver, &effective_min_version()) {
            return None;
        }

        let nodes_arr = obj.get("nodes")?.as_array()?;
        let weights_arr = obj.get("weights")?.as_array()?;

        if nodes_arr.len() != n || weights_arr.len() != n {
            return None;
        }
        let mut nodes = Vec::with_capacity(n);
        let mut weights = Vec::with_capacity(n);
        for s in nodes_arr {
            nodes.push(Float::with_val(prec, Float::parse(s.as_str()?).ok()?));
        }
        for s in weights_arr {
            weights.push(Float::with_val(prec, Float::parse(s.as_str()?).ok()?));
        }
        Some((nodes, weights))
    }

    /// Returns `true` if `a` is strictly older than `b` using simple
    /// major.minor.patch comparison (no semver pre-release handling
    /// needed — toolkit versions are always plain `X.Y.Z`).
    fn version_is_older(a: &str, b: &str) -> bool {
        let parse = |s: &str| -> (u64, u64, u64) {
            let mut parts = s.splitn(3, '.');
            let major = parts.next().and_then(|x| x.parse().ok()).unwrap_or(0);
            let minor = parts.next().and_then(|x| x.parse().ok()).unwrap_or(0);
            let patch = parts.next().and_then(|x| x.parse().ok()).unwrap_or(0);
            (major, minor, patch)
        };
        parse(a) < parse(b)
    }

    /// Print a diagnostic warning to stderr when a cache file exists
    /// but fails to load (corrupt JSON, malformed zip, structural
    /// identity violation). Includes the file path and a short reason
    /// so reviewers can identify and remediate corrupt fixtures
    /// without silently triggering a multi-minute Newton recompute.
    fn warn_cache_skip(path: &std::path::Path, reason: &str) {
        eprintln!(
            "[gl_cache] WARNING: skipping {} ({}); recomputing",
            path.display(),
            reason
        );
    }

    fn load_gl_cache(n: usize, prec: u32, mode: CacheMode) -> Option<(Vec<Float>, Vec<Float>)> {
        if mode == CacheMode::Off {
            return None;
        }

        // Caches are stored as .json.zip only — we read straight from the
        // zip (decompress in memory) and never write a decompressed .json.
        // This keeps local disk usage ~2x smaller so more configs stay
        // cached during sweeps. JsonOnly is now a no-op for reads (kept
        // for API compatibility) since no uncompressed .json is written.
        if mode == CacheMode::JsonOnly {
            return None;
        }

        // Local zip — decompress in memory.
        if let Some(result) = try_load_local_zip(n, prec) {
            return Some(result);
        }

        None
    }

    /// Attempt to load `(n, prec)` from a local `.json.zip`.
    /// Decompresses in memory; does NOT write a decompressed `.json`.
    /// Returns `None` if the zip is absent, corrupt, or structurally invalid.
    fn try_load_local_zip(n: usize, prec: u32) -> Option<(Vec<Float>, Vec<Float>)> {
        let zip_path = gl_cache_zip_path(n, prec)?;
        if !zip_path.exists() {
            return None;
        }
        match load_from_zip(&zip_path, n, prec) {
            Some((parsed, _json_string)) => {
                if let Some(reason) = cache_structural_check(&parsed.0, &parsed.1, prec) {
                    warn_cache_skip(&zip_path, &reason);
                    None
                } else {
                    Some(parsed)
                }
            }
            None => {
                warn_cache_skip(&zip_path, "zip open / decompress / shape parse failed");
                None
            }
        }
    }

    /// Test-only accessor for `parse_gl_json` (lets version-rejection
    /// tests call the parser directly without touching disk).
    #[cfg(test)]
    pub fn parse_gl_json_for_test(
        data: &str,
        n: usize,
        prec: u32,
    ) -> Option<(Vec<Float>, Vec<Float>)> {
        parse_gl_json(data, n, prec)
    }

    /// Read a zip cache file. Expects the archive to contain exactly one
    /// entry whose name matches the uncompressed JSON filename
    /// (`prec{prec}_npts{n}.json`).
    ///
    /// Returns the parsed `(nodes, weights)` plus the raw JSON string,
    /// so the caller can write the decompressed copy to disk without
    /// re-serializing.
    fn load_from_zip(zip_path: &std::path::Path, n: usize, prec: u32) -> Option<(GlTable, String)> {
        use std::io::Read;
        let file = std::fs::File::open(zip_path).ok()?;
        let mut archive = zip::ZipArchive::new(file).ok()?;
        if archive.len() != 1 {
            return None;
        }
        let entry_name = format!("prec{}_npts{}.json", prec, n);
        let mut entry = archive.by_name(&entry_name).ok()?;
        let mut data = String::new();
        entry.read_to_string(&mut data).ok()?;
        let parsed = parse_gl_json(&data, n, prec)?;
        Some((parsed, data))
    }

    fn save_gl_cache(n: usize, prec: u32, nodes: &[Float], weights: &[Float], mode: CacheMode) {
        // Off writes nothing. JsonOnly also writes nothing now: the cache
        // is zip-only (we never persist a decompressed .json), so the
        // only meaningful write mode is JsonZip.
        if matches!(mode, CacheMode::Off | CacheMode::JsonOnly) {
            return;
        }

        // Serialize to the versioned JSON envelope, then write ONLY the
        // deflated .json.zip. No uncompressed .json is written — readers
        // decompress from the zip on demand. This halves local disk use.
        if let Some(path) = gl_cache_path(n, prec) {
            let ns: Vec<String> = nodes.iter().map(|f| f.to_string()).collect();
            let ws: Vec<String> = weights.iter().map(|f| f.to_string()).collect();
            let json = serde_json::json!({
                "schema_version": 1,
                "toolkit_version": TOOLKIT_VERSION,
                "n_pts": n,
                "precision_bits": prec,
                "nodes": ns,
                "weights": ws,
            });
            let json_str = match serde_json::to_string(&json) {
                Ok(s) => s,
                Err(_) => return,
            };

            use std::io::Write;
            let entry_name = format!("prec{}_npts{}.json", prec, n);
            let zip_filename = format!("{}.zip", entry_name);
            let zip_path = match path.parent() {
                Some(p) => p.join(&zip_filename),
                None => return,
            };
            // large_file(true): the `zip` crate defaults to classic
            // (non-Zip64) headers, which silently abort the write once
            // either the uncompressed or compressed size crosses 4 GiB
            // (see xc_spectral::ccm::hp::tau_cache::compress_to_zip for
            // the full writeup of this failure mode). GL tables are
            // small in practice, but this keeps the write path uniform
            // and safe regardless of table size.
            let mut buf: Vec<u8> = Vec::with_capacity(json_str.len() / 2);
            {
                let cursor = std::io::Cursor::new(&mut buf);
                let mut writer = zip::ZipWriter::new(cursor);
                let opts: zip::write::SimpleFileOptions = zip::write::SimpleFileOptions::default()
                    .compression_method(zip::CompressionMethod::Deflated)
                    .large_file(true);
                if writer.start_file(&entry_name, opts).is_err() {
                    return;
                }
                if writer.write_all(json_str.as_bytes()).is_err() {
                    return;
                }
                if writer.finish().is_err() {
                    return;
                }
            }
            let _ = std::fs::write(&zip_path, &buf);
        }
    }

    #[cfg(test)]
    pub(super) fn gauss_legendre_compute(n: usize, prec: u32) -> (Vec<Float>, Vec<Float>) {
        gauss_legendre_compute_scheduled(n, prec, crate::hp_runtime::GlRootSchedule::serial())
    }

    fn gauss_legendre_compute_scheduled(
        n: usize,
        prec: u32,
        root_schedule: crate::hp_runtime::GlRootSchedule,
    ) -> (Vec<Float>, Vec<Float>) {
        let pi_v = pi(prec);
        let one = Float::with_val(prec, 1);
        let four_n_plus_two = fl_i(prec, (4 * n + 2) as i64);
        let eps_threshold = Float::with_val(prec, 2).pow(-((prec as i32) - 8));
        // Root construction remains serial by default. An owning HP batch may
        // opt into this single-table exception after selecting exactly one
        // parallel level. WSL remains excluded from supported qualification
        // because concurrent GMP allocation has failed there nondeterministically.
        let mut combined = if let Some(min_task_len) = root_schedule.parallel_min_task_len() {
            use rayon::prelude::*;
            (0..n)
                .into_par_iter()
                .with_min_len(min_task_len)
                .map(|offset| {
                    gauss_legendre_root(
                        n,
                        offset + 1,
                        prec,
                        &pi_v,
                        &one,
                        &four_n_plus_two,
                        &eps_threshold,
                    )
                })
                .collect::<Vec<_>>()
        } else {
            (1..=n)
                .map(|k| {
                    gauss_legendre_root(n, k, prec, &pi_v, &one, &four_n_plus_two, &eps_threshold)
                })
                .collect::<Vec<_>>()
        };
        combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        combined.into_iter().unzip()
    }

    fn gauss_legendre_root(
        n: usize,
        k: usize,
        prec: u32,
        pi_v: &Float,
        one: &Float,
        four_n_plus_two: &Float,
        eps_threshold: &Float,
    ) -> (Float, Float) {
        let mut phi = pi_v.clone();
        phi *= fl_i(prec, (4 * k - 1) as i64);
        phi /= four_n_plus_two;
        let mut x = phi.cos();
        for _ in 0..50 {
            let (pn, pn_prime) = legendre_p_and_deriv(n, &x, prec);
            let mut dx = pn;
            dx /= &pn_prime;
            x -= &dx;
            if dx
                .cmp_abs(eps_threshold)
                .map(|ordering| ordering.is_lt())
                .unwrap_or(false)
            {
                break;
            }
        }
        let (_pn, pn_prime) = legendre_p_and_deriv(n, &x, prec);
        let one_minus_x2 = {
            let mut value = one.clone();
            value -= x.clone().square();
            value
        };
        let mut denominator = one_minus_x2;
        denominator *= &pn_prime.square();
        let mut weight = Float::with_val(prec, 2);
        weight /= &denominator;
        (x, weight)
    }

    #[cfg(test)]
    pub(super) fn gauss_legendre_compute_allocating_reference(
        n: usize,
        prec: u32,
    ) -> (Vec<Float>, Vec<Float>) {
        let pi_v = pi(prec);
        let one = Float::with_val(prec, 1);
        let mut nodes = Vec::with_capacity(n);
        let mut weights = Vec::with_capacity(n);
        let four_n_plus_two = fl_i(prec, (4 * n + 2) as i64);
        for k in 1..=n {
            let mut phi = pi_v.clone();
            phi *= fl_i(prec, (4 * k - 1) as i64);
            phi /= &four_n_plus_two;
            let mut x = phi.cos();
            let eps_threshold = Float::with_val(prec, 2).pow(-((prec as i32) - 8));
            for _ in 0..50 {
                let (pn, pn_prime) = legendre_p_and_deriv(n, &x, prec);
                let mut dx = pn;
                dx /= &pn_prime;
                x -= &dx;
                if dx
                    .cmp_abs(&eps_threshold)
                    .map(|ordering| ordering.is_lt())
                    .unwrap_or(false)
                {
                    break;
                }
            }
            let (_pn, pn_prime) = legendre_p_and_deriv(n, &x, prec);
            let one_minus_x2 = {
                let mut value = one.clone();
                value -= x.clone().square();
                value
            };
            let mut denominator = one_minus_x2;
            denominator *= &pn_prime.square();
            let mut weight = Float::with_val(prec, 2);
            weight /= &denominator;
            nodes.push(x);
            weights.push(weight);
        }
        let mut combined: Vec<(Float, Float)> = nodes.into_iter().zip(weights).collect();
        combined.sort_by(|left, right| left.0.partial_cmp(&right.0).unwrap());
        let nodes = combined.iter().map(|pair| pair.0.clone()).collect();
        let weights = combined.iter().map(|pair| pair.1.clone()).collect();
        (nodes, weights)
    }

    fn legendre_p_and_deriv(n: usize, x: &Float, prec: u32) -> (Float, Float) {
        let one = Float::with_val(prec, 1);
        if n == 0 {
            return (one, Float::with_val(prec, 0));
        }
        let mut p0 = Float::with_val(prec, 1);
        let mut p1 = x.clone();
        if n == 1 {
            return (p1, one);
        }
        for k in 1..n {
            let kf = k as i64;
            let mut t1 = x.clone();
            t1 *= &p1;
            t1 *= fl_i(prec, 2 * kf + 1);
            let mut t2 = p0.clone();
            t2 *= fl_i(prec, kf);
            let mut p_next = t1;
            p_next -= &t2;
            p_next /= fl_i(prec, kf + 1);
            p0 = p1;
            p1 = p_next;
        }
        let nf = n as i64;
        let mut numer = x.clone();
        numer *= &p1;
        numer -= &p0;
        numer *= fl_i(prec, nf);
        let mut denom = x.clone().square();
        denom -= 1u32;
        let mut deriv = numer;
        deriv /= &denom;
        (p1, deriv)
    }

    // ===========================================================================
    // Public cache-verification API
    // ===========================================================================

    /// Per-file outcome from `verify_gl_cache_dir`.
    #[derive(Debug, Clone)]
    pub enum CacheFileStatus {
        /// File loaded and passed all structural identity checks.
        Ok {
            path: std::path::PathBuf,
            n: usize,
            prec: u32,
        },
        /// File path was not in the expected `prec{P}_npts{N}.json[.zip]`
        /// pattern; skipped.
        Skipped {
            path: std::path::PathBuf,
            reason: String,
        },
        /// File was in the expected pattern but failed to load (malformed
        /// JSON, malformed zip, IO error).
        LoadFailed {
            path: std::path::PathBuf,
            n: usize,
            prec: u32,
            reason: String,
        },
        /// File loaded successfully but failed at least one of the GL
        /// structural identities (Σw=2, Σx·w=0, antisymmetry).
        StructurallyInvalid {
            path: std::path::PathBuf,
            n: usize,
            prec: u32,
            reason: String,
        },
    }

    /// Aggregate report from `verify_gl_cache_dir`.
    #[derive(Debug, Clone)]
    pub struct CacheVerifyReport {
        /// Directory that was scanned.
        pub directory: std::path::PathBuf,
        /// One status entry per file found in `directory` (in arbitrary
        /// filesystem order). Files outside the expected naming pattern
        /// are reported as `CacheFileStatus::Skipped`.
        pub statuses: Vec<CacheFileStatus>,
    }

    impl CacheVerifyReport {
        /// Count of files that passed all checks.
        pub fn ok_count(&self) -> usize {
            self.statuses
                .iter()
                .filter(|s| matches!(s, CacheFileStatus::Ok { .. }))
                .count()
        }
        /// Count of files that failed at least one check (load or
        /// structural). Skipped files are not counted as failures.
        pub fn failure_count(&self) -> usize {
            self.statuses
                .iter()
                .filter(|s| {
                    matches!(
                        s,
                        CacheFileStatus::LoadFailed { .. }
                            | CacheFileStatus::StructurallyInvalid { .. }
                    )
                })
                .count()
        }
        /// All failure entries (load + structural), for callers that
        /// want to print only the bad files.
        pub fn failures(&self) -> impl Iterator<Item = &CacheFileStatus> {
            self.statuses.iter().filter(|s| {
                matches!(
                    s,
                    CacheFileStatus::LoadFailed { .. }
                        | CacheFileStatus::StructurallyInvalid { .. }
                )
            })
        }
    }

    /// Parse a cache filename of the form `prec{P}_npts{N}.json` or
    /// `prec{P}_npts{N}.json.zip`, returning `(N, P)`. Returns `None`
    /// for any other filename pattern.
    fn parse_cache_filename(name: &str) -> Option<(usize, u32)> {
        let stem = name
            .strip_suffix(".json.zip")
            .or_else(|| name.strip_suffix(".json"))?;
        // stem now: "prec{P}_npts{N}"
        let after_prec = stem.strip_prefix("prec")?;
        let (prec_str, n_str) = after_prec.split_once("_npts")?;

        let prec: u32 = prec_str.parse().ok()?;
        let n: usize = n_str.parse().ok()?;
        Some((n, prec))
    }

    /// Walk the given cache directory and structurally verify every
    /// `prec{P}_npts{N}.json[.zip]` file in it. Returns a per-file
    /// status report; does not mutate any files (corrupt files are
    /// not deleted).
    ///
    /// Use this from a CLI wrapper to audit a cache directory before
    /// a long HP run, e.g.:
    ///
    /// ```text
    /// let report = verify_gl_cache_dir(std::path::Path::new("data/gl_cache"))?;
    /// for failure in report.failures() { eprintln!("{:?}", failure); }
    /// if report.failure_count() > 0 { std::process::exit(1); }
    /// ```
    ///
    /// The verification runs sequentially per file. At HP-3338 with
    /// thousands of cache files this can take several seconds; for
    /// production use callers may want to prune the directory first
    /// to the precisions they actually plan to use.
    pub fn verify_gl_cache_dir(dir: &std::path::Path) -> std::io::Result<CacheVerifyReport> {
        let mut statuses: Vec<CacheFileStatus> = Vec::new();

        if !dir.exists() {
            return Ok(CacheVerifyReport {
                directory: dir.to_path_buf(),
                statuses,
            });
        }

        let entries = std::fs::read_dir(dir)?;
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let name = match path.file_name().and_then(|s| s.to_str()) {
                Some(n) => n,
                None => continue,
            };

            let (n, prec) = match parse_cache_filename(name) {
                Some(np) => np,
                None => {
                    statuses.push(CacheFileStatus::Skipped {
                        path: path.clone(),
                        reason: format!(
                            "filename '{}' not in expected prec{{P}}_npts{{N}}.json[.zip] form",
                            name
                        ),
                    });
                    continue;
                }
            };

            // Load the file.
            let parsed: Option<(Vec<Float>, Vec<Float>)> = if name.ends_with(".json.zip") {
                load_from_zip(&path, n, prec).map(|(p, _)| p)
            } else {
                std::fs::read_to_string(&path)
                    .ok()
                    .and_then(|data| parse_gl_json(&data, n, prec))
            };

            let (nodes, weights) = match parsed {
                Some(p) => p,
                None => {
                    statuses.push(CacheFileStatus::LoadFailed {
                        path: path.clone(),
                        n,
                        prec,
                        reason: "parse / decompress failed".to_string(),
                    });
                    continue;
                }
            };

            // Structural identity check.
            match cache_structural_check(&nodes, &weights, prec) {
                None => {
                    statuses.push(CacheFileStatus::Ok { path, n, prec });
                }
                Some(reason) => {
                    statuses.push(CacheFileStatus::StructurallyInvalid {
                        path,
                        n,
                        prec,
                        reason,
                    });
                }
            }
        }

        Ok(CacheVerifyReport {
            directory: dir.to_path_buf(),
            statuses,
        })
    }
}

#[cfg(feature = "hp")]
pub use hp::{
    gauss_legendre_nodes, gauss_legendre_nodes_scheduled, gauss_legendre_nodes_via_cache,
    gauss_legendre_nodes_via_cache_scheduled, verify_gl_cache_dir, CacheFileStatus, CacheMode,
    CacheVerifyReport, CachedQuadratureRule, QuadratureCacheRequest,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "hp")]
    #[test]
    fn hp_quadrature_uses_common_cache_fabric_for_round_trip() {
        use xc_cache::{
            ArtifactExecutionCacheMode, CacheLayer, CachePolicy, CacheQuality, CacheResolver,
            CacheVisibility, CertificationFailurePolicy, FilesystemCacheStore,
            ManagedArtifactCacheConfig, ManagedArtifactCacheSession, ManagedRemoteCacheMode,
            ManagedRunProfile, OutputPreservationValidationReport, OutputValidationConfig,
            ToolkitVersion,
        };
        use xc_core::{CacheLookupOutcome, CacheReuseDisposition};

        let root = std::env::temp_dir().join(format!(
            "xc-numerics-quadrature-fabric-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let reference_root = root.join("reference");
        let resolver = CacheResolver::new(vec![CacheLayer {
            precedence: 0,
            store: Box::new(FilesystemCacheStore::new(
                "workstation",
                &reference_root,
                true,
                CacheVisibility::Local,
            )),
        }]);
        let policy = CachePolicy {
            current_toolkit_version: ToolkitVersion::parse("0.13.0").unwrap(),
            minimum_quality: CacheQuality::Validated,
            accepted_schema_versions: vec![1],
            allow_deprecated: false,
            allow_quarantined: false,
            allowed_visibilities: vec![CacheVisibility::Local],
        };
        let request = || QuadratureCacheRequest {
            resolver: Some(&resolver),
            reference_resolver: None,
            acceptance: Some(&policy),
            ordered_overlays: vec!["workstation".to_owned()],
            mode: ArtifactExecutionCacheMode::PreferReuse,
            write_on_miss: true,
            write_visibility: CacheVisibility::Local,
            requested_assurance: xc_core::AssuranceLevel::Computed,
            certification_failure_policy:
                xc_cache::CertificationFailurePolicy::RetainComputedFailRun,
            production_sink: None,
        };
        let first = gauss_legendre_nodes_via_cache(4, 128, request()).unwrap();
        let second = gauss_legendre_nodes_via_cache(4, 128, request()).unwrap();
        assert_eq!(first.cache_access.lookup_outcome, CacheLookupOutcome::Miss);
        assert_eq!(second.cache_access.lookup_outcome, CacheLookupOutcome::Hit);
        assert_eq!(
            second.cache_access.reuse_disposition,
            CacheReuseDisposition::Reused
        );
        assert_eq!(first.nodes, second.nodes);
        assert_eq!(first.weights, second.weights);

        let validation_root = root.join("validation");
        let session = ManagedArtifactCacheSession::with_layers_for_test(
            ManagedArtifactCacheConfig {
                profile: ManagedRunProfile::Normal,
                requested_assurance: xc_core::AssuranceLevel::Computed,
                certification_failure_policy: CertificationFailurePolicy::RetainComputedFailRun,
                cache_root: root.join("production"),
                staging_root: None,
                publication_target: xc_core::PublicationTarget::None,
                repository_owner: "local-fixture".to_owned(),
                remote_cache_mode: ManagedRemoteCacheMode::None,
                cache_mode: ArtifactExecutionCacheMode::VerifyAgainstReference,
                replace_existing_publication: false,
                execute_remote_mutations: false,
                output_validation: Some(OutputValidationConfig {
                    validation_root: validation_root.clone(),
                    report_root: validation_root.join("reports"),
                    reference_mode: ManagedRemoteCacheMode::Public,
                }),
            },
            vec![CacheLayer {
                precedence: 0,
                store: Box::new(FilesystemCacheStore::new(
                    "validation-computed",
                    validation_root.join("computed"),
                    true,
                    CacheVisibility::Local,
                )),
            }],
            vec![CacheLayer {
                precedence: 0,
                store: Box::new(FilesystemCacheStore::new(
                    "reference",
                    &reference_root,
                    false,
                    CacheVisibility::Local,
                )),
            }],
        )
        .unwrap();
        let verified = gauss_legendre_nodes_via_cache(4, 128, session.context()).unwrap();
        assert_eq!(verified.nodes, first.nodes);
        assert_eq!(verified.weights, first.weights);
        session.finalize_publication_inventory().unwrap();
        let report: OutputPreservationValidationReport = serde_json::from_slice(
            &std::fs::read(validation_root.join("reports/latest.json")).unwrap(),
        )
        .unwrap();
        assert!(report.output_preserving);
        assert_eq!(report.totals.matched, 1);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn quadrature_reuse_plan_is_machine_readable() {
        let plan = quadrature_artifact_reuse_plan();
        plan.validate().unwrap();
        assert!(plan
            .artifacts
            .iter()
            .any(|node| node.kind == "rule_nodes_weights"));
    }

    /// GL-64 should integrate x² on [0, 1] exactly (polynomial degree < 2*64).
    #[test]
    fn gl64_integrates_x_squared() {
        let result = gauss_legendre_64pt_f64(|x| x * x, 0.0, 1.0);
        let expected = 1.0 / 3.0;
        let rel_err = (result - expected).abs() / expected;
        assert!(
            rel_err < 1e-14,
            "GL-64 x² integral: got {}, expected {}, rel err {:.2e}",
            result,
            expected,
            rel_err
        );
    }

    /// GL-64 should integrate sin(x) on [0, π] to high accuracy.
    #[test]
    fn gl64_integrates_sin() {
        let result = gauss_legendre_64pt_f64(|x| x.sin(), 0.0, std::f64::consts::PI);
        let expected = 2.0;
        let rel_err = (result - expected).abs() / expected;
        assert!(
            rel_err < 1e-13,
            "GL-64 sin integral: got {}, expected {}, rel err {:.2e}",
            result,
            expected,
            rel_err
        );
    }

    /// GL-64 should integrate exp(-x²) on [-5, 5] ≈ √π.
    #[test]
    fn gl64_integrates_gaussian() {
        let result = gauss_legendre_64pt_f64(|x| (-x * x).exp(), -5.0, 5.0);
        let expected = std::f64::consts::PI.sqrt();
        let rel_err = (result - expected).abs() / expected;
        assert!(
            rel_err < 1e-10,
            "GL-64 Gaussian integral: got {}, expected {}, rel err {:.2e}",
            result,
            expected,
            rel_err
        );
    }

    /// `gauss_legendre_npt_f64` at N=8 should integrate any polynomial
    /// of degree < 2N = 16 exactly. We test x^15 on [0, 1]:
    /// ∫₀¹ x¹⁵ dx = 1/16.
    #[test]
    fn gl_npt_integrates_polynomial_exactly_n8() {
        let result = gauss_legendre_npt_f64(|x| x.powi(15), 0.0, 1.0, 8);
        let expected = 1.0 / 16.0;
        let abs_err = (result - expected).abs();
        // GL with N nodes is exact (modulo f64 rounding) for polynomials
        // of degree ≤ 2N-1.
        assert!(
            abs_err < 1e-14,
            "GL-8 ∫x¹⁵ on [0,1]: got {}, expected {}, abs err {:.2e}",
            result,
            expected,
            abs_err
        );
    }

    /// `gauss_legendre_npt_f64` at N=4 should integrate ∫₀¹ x⁷ = 1/8
    /// exactly (degree 7 < 2·4 = 8).
    #[test]
    fn gl_npt_n4_polynomial_exact() {
        let result = gauss_legendre_npt_f64(|x| x.powi(7), 0.0, 1.0, 4);
        let expected = 1.0 / 8.0;
        let abs_err = (result - expected).abs();
        assert!(
            abs_err < 1e-14,
            "GL-4 ∫x⁷ on [0,1]: got {}, expected {}, abs err {:.2e}",
            result,
            expected,
            abs_err
        );
    }

    /// `gauss_legendre_npt_f64` should fail to integrate degree 2N exactly:
    /// at N=4 (max exact degree 7), degree 8 has nonzero error.
    /// We don't need the error to be a specific value — just that it's
    /// detectably nonzero (vs the polynomial-exact case above).
    #[test]
    fn gl_npt_above_exact_degree_has_error() {
        // ∫₀¹ x⁸ = 1/9
        let result = gauss_legendre_npt_f64(|x| x.powi(8), 0.0, 1.0, 4);
        let expected = 1.0 / 9.0;
        let abs_err = (result - expected).abs();
        // Error at N=4 for degree 2N=8 is O(10⁻⁵) for x⁸ on [0,1].
        // We just check it's larger than the polynomial-exact case (~1e-14).
        assert!(
            abs_err > 1e-7,
            "GL-4 ∫x⁸ should have appreciable error (degree exceeds 2N-1); got {:.2e}",
            abs_err
        );
    }

    #[cfg(feature = "hp")]
    #[test]
    fn scheduled_gl_roots_are_exactly_identical_across_worker_counts() {
        use crate::hp_runtime::GlRootSchedule;

        let one_worker = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let four_workers = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        for (order, precision) in [(8, 256), (8, 3386), (8, 6708)] {
            let serial = gauss_legendre_nodes(order, precision, CacheMode::Off);
            let one = one_worker.install(|| {
                gauss_legendre_nodes_scheduled(
                    order,
                    precision,
                    CacheMode::Off,
                    GlRootSchedule::parallel_for_test(1),
                )
            });
            let four = four_workers.install(|| {
                gauss_legendre_nodes_scheduled(
                    order,
                    precision,
                    CacheMode::Off,
                    GlRootSchedule::parallel_for_test(1),
                )
            });

            let serialize = |table: &(Vec<rug::Float>, Vec<rug::Float>)| {
                (
                    table
                        .0
                        .iter()
                        .map(rug::Float::to_string)
                        .collect::<Vec<_>>(),
                    table
                        .1
                        .iter()
                        .map(rug::Float::to_string)
                        .collect::<Vec<_>>(),
                )
            };
            assert_eq!(serialize(&one), serialize(&serial));
            assert_eq!(serialize(&four), serialize(&serial));
        }
    }
}

#[cfg(all(test, feature = "hp"))]
mod hp_cache_tests {
    //! Tests for the HP GL cache lookup logic introduced in v0.4.1.
    //!
    //! Cache lookup tests use a thread-local test root. They neither depend
    //! on nor mutate XC_CACHE_ROOT or the process working directory, and can
    //! safely execute concurrently with numerical tests.

    use super::*;
    use rug::Float;
    use std::io::Write;
    use std::path::PathBuf;

    #[test]
    fn allocation_reduction_preserves_exact_gl_payload_values() {
        for (order, precision) in [(1, 128), (8, 192), (33, 257)] {
            let reference = hp::gauss_legendre_compute_allocating_reference(order, precision);
            let actual = hp::gauss_legendre_compute(order, precision);
            assert_eq!(
                actual, reference,
                "GL values changed at n={order}, p={precision}"
            );
        }
    }

    /// A panic-safe per-thread override, independent of the user's cache
    /// environment. No global cwd mutation is necessary for cache tests.
    struct CacheRootGuard {
        original: Option<PathBuf>,
    }
    impl CacheRootGuard {
        fn enter(temp: &std::path::Path) -> Self {
            Self {
                original: hp::replace_test_cache_root(Some(temp.join("data"))),
            }
        }
    }
    impl Drop for CacheRootGuard {
        fn drop(&mut self) {
            hp::replace_test_cache_root(self.original.take());
        }
    }

    /// Make a fresh, unique throwaway directory under the workspace
    /// `target/test-tmp/` dir (not the OS temp dir). Keeping test scratch
    /// inside `target/` means it is contained in the repo's build area
    /// and removed by `cargo clean`, rather than scattering directories
    /// in `/tmp` or `%TEMP%`. The path is resolved from
    /// `CARGO_MANIFEST_DIR` at compile time, so it is correct regardless
    /// of the process's runtime cwd. A tag plus a process-id +
    /// nanosecond suffix avoids clashes when tests run in parallel or
    /// are re-run rapidly.
    fn fresh_temp_dir(tag: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let pid = std::process::id();
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("target")
            .join("test-tmp")
            .join(format!(
                "xc_numerics_gl_cache_test_{}_{}_{}",
                tag, pid, nanos
            ));
        std::fs::create_dir_all(&dir).expect("create test tmp dir");
        dir
    }

    /// Write a 2-element JSON array of decimal-string node/weight
    /// values for the given (n, prec). The values are deterministic
    /// fakes — not actual GL nodes, but well-formed enough to round
    /// trip through `parse_gl_json`.
    /// Generate a structurally-valid synthetic GL cache JSON for use in
    /// cache-priority and zip-fallback tests, where the *content* of
    /// the cache file isn't important (we're testing the cache lookup
    /// machinery, not GL correctness). The values are not real GL
    /// nodes/weights, but they satisfy the three structural identities
    /// `cache_structural_check` enforces:
    ///
    ///   - antisymmetric nodes:  nodes[i] + nodes[n-1-i] = 0
    ///   - mirror weights:       weights[i] = weights[n-1-i]
    ///   - sum of weights = 2,   first moment = 0
    ///
    /// **Exactness matters.** The structural validator runs at HP
    /// precision and computes Σw exactly. If the f64-formatted weight
    /// strings round-trip through HP with ULP error (e.g. "4e-01" for
    /// 2/5, which is 0.4 + 2e-17 in binary), Σw is not exactly 2 and
    /// the validator (correctly) rejects the fixture. We side-step this
    /// by picking weights from `{1/4, 1/2, 3/4, 1/8, ...}` — exact
    /// powers-of-2 fractions that round-trip through binary HP exactly.
    /// For each `n`, the weight pattern is chosen so the sum is
    /// exactly 2 in binary arithmetic.
    ///
    /// `epsilon` controls the smallest absolute node value, so two
    /// synthetic fixtures with different epsilons are distinguishable
    /// (used by the cache-priority test).
    fn synthetic_valid_gl_json(n: usize, epsilon: f64) -> String {
        // Weight pattern: pick mirror-symmetric weights summing to 2
        // using only exact-binary fractions (1/2, 1/4, 3/4, 3/8, etc.).
        // For arbitrary n, use a uniform "{1/2 each pair}" structure
        // when n is even and `2/n` is exact (power of 2), and a
        // "split-the-middle" structure otherwise.
        //
        // The simplest universal pattern: weights = [w, w, ..., w_center, ..., w, w]
        // with paired w = 1/2 and w_center = 2 - n_pairs (for odd n only).
        //
        // For n even with an even count of pairs:
        //   each weight = 2/n. Exact only if n ∈ {1, 2, 4, 8, 16}.
        //
        // We instead use a deterministic recipe that's exact for any n:
        //   - For n = 1: [2.0]   (degenerate; rarely used in tests)
        //   - For n = 2: [1.0, 1.0]
        //   - For n ≥ 3: pair the first n-2 elements with weight 1/2 each
        //                (sum = (n-2)/2 from these), and split the
        //                remaining (2 - (n-2)/2) = (6 - n)/2 = ?
        //
        // Cleaner recipe: assign weight 0.5 to indices {0, n-1, 1, n-2}
        // (the outer two pairs) and the remainder uniformly to the rest.
        // But "the rest" weight = (2 - 4*0.5) / (n-4) = 0/(n-4) = 0 for
        // n > 4 — which makes interior weights zero (technically
        // structurally valid: zero is symmetric and contributes nothing).
        //
        // Use this: outer two pairs each weighted 0.5 (mirror); all
        // interior weights = 0. Σw = 4 × 0.5 = 2 exactly. Σx·w on the
        // antisymmetric outer four cancels exactly. The interior zero
        // weights contribute 0 to all moments. Structurally valid for
        // any n ≥ 4.
        //
        // For n < 4: special-case. n = 2: both 1.0; n = 3: outer pair
        // 0.5 each, center 1.0.
        let weights_exact: Vec<&'static str> = match n {
            0 => Vec::new(),
            1 => vec!["2"],
            2 => vec!["1", "1"],
            3 => vec!["5e-1", "1", "5e-1"],
            _ => {
                // n ≥ 4: outer two pairs at 0.5, interior zero.
                let mut w = vec!["0"; n];
                w[0] = "5e-1";
                w[n - 1] = "5e-1";
                w[1] = "5e-1";
                w[n - 2] = "5e-1";
                w
            }
        };

        // Nodes: antisymmetric. We use simple f64 arithmetic to build
        // them, then format the strings; mirror-symmetry of the
        // pre/post-image ensures that even when individual values have
        // ULP error in their f64 representation, the antisymmetric sum
        // (nodes[i] + nodes[n-1-i]) of two strings parsed independently
        // at HP gives exactly 0 (since one parsed value is the negation
        // of the other, character-for-character).
        let nodes: Vec<String> = (0..n)
            .map(|i| {
                match (2 * i + 1).cmp(&n) {
                    std::cmp::Ordering::Less => {
                        // Lower half.
                        let v = epsilon + 0.1 * (i as f64);
                        format!("-{:.20e}", v)
                    }
                    std::cmp::Ordering::Equal => {
                        // Center (only for odd n).
                        "0".to_string()
                    }
                    std::cmp::Ordering::Greater => {
                        // Upper half: mirror of lower half.
                        let j = n - 1 - i;
                        let v = epsilon + 0.1 * (j as f64);
                        format!("{:.20e}", v)
                    }
                }
            })
            .collect();

        // Weights are already exact decimal strings.
        let weights: Vec<String> = weights_exact.iter().map(|s| s.to_string()).collect();

        serde_json::json!({
            "schema_version": 1,
            "toolkit_version": hp::toolkit_version_for_test(),
            "n_pts": n,
            "precision_bits": 64,
            "nodes": nodes,
            "weights": weights,
        })
        .to_string()
    }

    /// Produce a valid-envelope JSON but with structurally-invalid
    /// nodes/weights (all zeros — Σw ≠ 2, nodes not antisymmetric).
    /// The parser accepts the envelope; the structural check rejects it.
    fn structurally_invalid_gl_json(n: usize, prec: u32) -> String {
        let zeros: Vec<String> = (0..n).map(|_| "0".to_string()).collect();
        serde_json::json!({
            "schema_version": 1,
            "toolkit_version": hp::toolkit_version_for_test(),
            "n_pts": n,
            "precision_bits": prec,
            "nodes": zeros.clone(),
            "weights": zeros,
        })
        .to_string()
    }

    /// Round-trip helper: compute or load nodes/weights, then verify
    /// that we got back exactly `n` of each at the requested
    /// precision, with no NaN.
    fn assert_well_formed(n: usize, prec: u32, nodes: &[Float], weights: &[Float]) {
        assert_eq!(nodes.len(), n, "node count");
        assert_eq!(weights.len(), n, "weight count");
        for (i, x) in nodes.iter().enumerate() {
            assert_eq!(x.prec(), prec, "node {} precision", i);
            assert!(!x.is_nan(), "node {} is NaN", i);
        }
        for (i, w) in weights.iter().enumerate() {
            assert_eq!(w.prec(), prec, "weight {} precision", i);
            assert!(!w.is_nan(), "weight {} is NaN", i);
        }
    }

    /// Zip-only contract: the `.json.zip` is the sole source of truth.
    /// Even if a stale uncompressed `.json` is present, it is ignored —
    /// the toolkit reads only the zip. (Tier-1 `.json` reads were
    /// removed when the cache became zip-only to halve disk usage.)
    #[test]
    fn cache_reads_zip_ignoring_stale_json() {
        let temp = fresh_temp_dir("zip_is_truth");
        let _guard = CacheRootGuard::enter(&temp);

        let n = 4;
        let prec: u32 = 64;
        let cache_dir = temp.join("data").join("gl_cache");
        std::fs::create_dir_all(&cache_dir).unwrap();

        // Stale .json fixture: epsilon = 0.01 (smallest |node| ≈ 0.01).
        // This must be IGNORED under the zip-only contract.
        let json_payload = synthetic_valid_gl_json(n, 0.01);
        let json_path = cache_dir.join(format!("prec{}_npts{}.json", prec, n));
        std::fs::write(&json_path, &json_payload).unwrap();

        // .zip fixture: epsilon = 0.5 (smallest |node| ≈ 0.5).
        // This is the source of truth and must win.
        let zip_path = cache_dir.join(format!("prec{}_npts{}.json.zip", prec, n));
        let zip_file = std::fs::File::create(&zip_path).unwrap();
        let mut zip_writer = zip::ZipWriter::new(zip_file);
        let opts: zip::write::SimpleFileOptions = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);
        zip_writer
            .start_file(format!("prec{}_npts{}.json", prec, n), opts)
            .unwrap();
        let zip_payload = synthetic_valid_gl_json(n, 0.5);
        zip_writer.write_all(zip_payload.as_bytes()).unwrap();
        zip_writer.finish().unwrap();

        let (nodes, weights) = hp::gauss_legendre_nodes(n, prec, hp::CacheMode::JsonZip);
        assert_well_formed(n, prec, &nodes, &weights);

        // The smallest |node| should match the .ZIP's epsilon (~0.5),
        // proving the zip was read and the stale .json was ignored.
        let mut smallest_abs = f64::INFINITY;
        for x in &nodes {
            let a = x.clone().abs().to_f64();
            if a < smallest_abs {
                smallest_abs = a;
            }
        }
        assert!(
            smallest_abs > 0.3,
            "expected smallest |node| ~0.5 from .zip fixture (zip-only contract), got {}",
            smallest_abs
        );
    }

    /// Zip fallback test: when only `.json.zip` exists, the toolkit
    /// must read it and decompress in-memory, WITHOUT writing a
    /// decompressed `.json` next to it (zip-only contract).
    #[test]
    fn cache_reads_zip_without_writing_decompressed_json() {
        let temp = fresh_temp_dir("zip_fallback");
        let _guard = CacheRootGuard::enter(&temp);

        let n = 5;
        let prec: u32 = 64;
        let cache_dir = temp.join("data").join("gl_cache");
        std::fs::create_dir_all(&cache_dir).unwrap();

        // Structurally-valid .json.zip with a known payload; no
        // uncompressed .json yet.
        let payload = synthetic_valid_gl_json(n, 0.05);
        let zip_path = cache_dir.join(format!("prec{}_npts{}.json.zip", prec, n));
        let zip_file = std::fs::File::create(&zip_path).unwrap();
        let mut zip_writer = zip::ZipWriter::new(zip_file);
        let opts: zip::write::SimpleFileOptions = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);
        zip_writer
            .start_file(format!("prec{}_npts{}.json", prec, n), opts)
            .unwrap();
        zip_writer.write_all(payload.as_bytes()).unwrap();
        zip_writer.finish().unwrap();

        let json_path = cache_dir.join(format!("prec{}_npts{}.json", prec, n));
        assert!(
            !json_path.exists(),
            "uncompressed .json should not exist before read"
        );

        let (nodes, weights) = hp::gauss_legendre_nodes(n, prec, hp::CacheMode::JsonZip);
        assert_well_formed(n, prec, &nodes, &weights);

        // Zip-only contract: reading from the .json.zip must NOT write a
        // decompressed .json — the zip is read in-memory each time.
        assert!(
            !json_path.exists(),
            "zip-only: no decompressed .json should be written after read"
        );
    }

    /// Compute-and-cache test: when neither `.json` nor `.json.zip`
    /// exists, the toolkit must compute fresh and write the result
    /// to the `.json.zip` path (zip-only — no uncompressed `.json`).
    #[test]
    fn cache_computes_fresh_and_writes_json() {
        let temp = fresh_temp_dir("compute_fresh");
        let _guard = CacheRootGuard::enter(&temp);

        let n = 8;
        let prec: u32 = 128;
        // Note: we deliberately do NOT create data/gl_cache/ ahead of
        // time — the toolkit must create the directory on demand.

        let json_path = temp
            .join("data")
            .join("gl_cache")
            .join(format!("prec{}_npts{}.json", prec, n));
        let zip_path = temp
            .join("data")
            .join("gl_cache")
            .join(format!("prec{}_npts{}.json.zip", prec, n));
        assert!(
            !json_path.exists(),
            "cache file should not exist before compute"
        );
        assert!(
            !zip_path.exists(),
            "cache zip should not exist before compute"
        );

        let (nodes, weights) = hp::gauss_legendre_nodes(n, prec, hp::CacheMode::JsonZip);
        assert_well_formed(n, prec, &nodes, &weights);

        // Zip-only: fresh compute writes the .json.zip, never the .json.
        assert!(
            zip_path.exists(),
            "fresh compute should write the .json.zip to <cwd>/data/gl_cache/..."
        );
        assert!(
            !json_path.exists(),
            "zip-only: fresh compute must not write an uncompressed .json"
        );

        // Sanity: cached value should round-trip. Re-reading should
        // not recompute (fast path), and we should get back the same
        // nodes/weights bit-for-bit.
        let (nodes2, weights2) = hp::gauss_legendre_nodes(n, prec, hp::CacheMode::JsonZip);
        assert_eq!(nodes.len(), nodes2.len());
        for (a, b) in nodes.iter().zip(nodes2.iter()) {
            // Compare via string form to avoid HP equality subtleties:
            // re-parsed Float should have the same string representation.
            assert_eq!(a.to_string(), b.to_string(), "node round-trip");
        }
        for (a, b) in weights.iter().zip(weights2.iter()) {
            assert_eq!(a.to_string(), b.to_string(), "weight round-trip");
        }
    }

    /// `CacheMode::Off` must never read or write any cache file: a fresh
    /// compute leaves the cache dir empty, and a pre-existing `.json` is
    /// ignored (recomputed, not read).
    #[test]
    fn cache_mode_off_never_touches_disk() {
        let temp = fresh_temp_dir("mode_off");
        let _guard = CacheRootGuard::enter(&temp);

        let n = 8;
        let prec: u32 = 128;
        let json_path = temp
            .join("data")
            .join("gl_cache")
            .join(format!("prec{}_npts{}.json", prec, n));
        let zip_path = temp
            .join("data")
            .join("gl_cache")
            .join(format!("prec{}_npts{}.json.zip", prec, n));

        let (nodes, weights) = hp::gauss_legendre_nodes(n, prec, hp::CacheMode::Off);
        assert_well_formed(n, prec, &nodes, &weights);

        // Off writes nothing.
        assert!(!json_path.exists(), "Off mode must not write .json");
        assert!(!zip_path.exists(), "Off mode must not write .json.zip");
    }

    /// `CacheMode::JsonOnly` must read a local `.json` but must NOT
    /// consult a `.json.zip`. We plant only a `.json.zip` (no `.json`)
    /// with a recognizable payload; JsonOnly should ignore it and
    /// recompute, leaving the real values (which pass structural checks),
    /// and crucially must NOT write a decompressed `.json` from the zip.
    #[test]
    fn cache_mode_json_only_ignores_zip() {
        let temp = fresh_temp_dir("mode_json_only");
        let _guard = CacheRootGuard::enter(&temp);

        let n = 4;
        let prec: u32 = 64;
        let cache_dir = temp.join("data").join("gl_cache");
        std::fs::create_dir_all(&cache_dir).unwrap();

        // Plant a structurally-bogus .json.zip (all-0.01 nodes/weights):
        // if JsonOnly wrongly consulted it, the result would differ from
        // a real GL-4 table. Build the zip with the canonical entry name.
        let zip_path = cache_dir.join(format!("prec{}_npts{}.json.zip", prec, n));
        {
            use std::io::Write;
            let ns: Vec<String> = (0..n).map(|_| "0.01".to_string()).collect();
            let ws: Vec<String> = (0..n).map(|_| "0.01".to_string()).collect();
            let payload = serde_json::json!([ns, ws]).to_string();
            let f = std::fs::File::create(&zip_path).unwrap();
            let mut zw = zip::ZipWriter::new(f);
            let opts: zip::write::SimpleFileOptions = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated);
            zw.start_file(format!("prec{}_npts{}.json", prec, n), opts)
                .unwrap();
            zw.write_all(payload.as_bytes()).unwrap();
            zw.finish().unwrap();
        }

        let json_path = cache_dir.join(format!("prec{}_npts{}.json", prec, n));
        assert!(!json_path.exists(), "no .json should exist before the call");

        // JsonOnly: must ignore the zip, recompute real values.
        let (nodes, weights) = hp::gauss_legendre_nodes(n, prec, hp::CacheMode::JsonOnly);
        assert_well_formed(n, prec, &nodes, &weights);

        // The smallest |node| of a real GL-4 table is ~0.339, NOT the
        // planted 0.01 — proving the zip was not consulted.
        let smallest = nodes
            .iter()
            .map(|x| x.clone().abs())
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let bogus = Float::with_val(prec, Float::parse("0.02").unwrap());
        assert!(
            smallest > bogus,
            "JsonOnly must not have used the planted zip (smallest |node| = {})",
            smallest
        );

        // Zip-only contract: JsonOnly is now a read/write no-op. It must
        // not consult the zip and must not write any .json.
        assert!(
            !json_path.exists(),
            "zip-only: JsonOnly must not write a decompressed .json"
        );
    }

    /// A GL cache file whose `toolkit_version` is older than the current
    /// the quadrature family producer floor must be rejected (returns `None` from
    /// the parser so the caller falls through to recompute). This
    /// simulates loading a stale file written by an older toolkit build
    /// whose output is no longer trusted.
    #[test]
    fn gl_cache_rejects_stale_toolkit_version() {
        let n = 4;
        let prec: u32 = 64;
        // Build a well-formed envelope but stamp a far-past toolkit_version.
        let ns: Vec<String> = (0..n).map(|i| format!("{}", i)).collect();
        let ws: Vec<String> = (0..n).map(|i| format!("{}", i)).collect();
        let payload = serde_json::json!({
            "toolkit_version": "0.0.1",
            "n_pts": n,
            "precision_bits": prec,
            "nodes": ns,
            "weights": ws,
        })
        .to_string();
        // parse_gl_json must return None because 0.0.1 is below the family floor.
        assert!(
            hp::parse_gl_json_for_test(&payload, n, prec).is_none(),
            "parser should reject a stale cache file with toolkit_version=0.0.1"
        );
    }

    /// Zip-only contract: fresh compute writes ONLY the `.json.zip`
    /// (no uncompressed `.json`). Readers decompress in-memory on demand.
    #[test]
    fn cache_fresh_compute_writes_zip_only() {
        let temp = fresh_temp_dir("compute_writes_zip_only");
        let _guard = CacheRootGuard::enter(&temp);

        let n = 8;
        let prec: u32 = 128;

        let json_path = temp
            .join("data")
            .join("gl_cache")
            .join(format!("prec{}_npts{}.json", prec, n));
        let zip_path = temp
            .join("data")
            .join("gl_cache")
            .join(format!("prec{}_npts{}.json.zip", prec, n));
        assert!(!json_path.exists(), ".json should not exist before compute");
        assert!(
            !zip_path.exists(),
            ".json.zip should not exist before compute"
        );

        // Trigger fresh compute.
        let (nodes, weights) = hp::gauss_legendre_nodes(n, prec, hp::CacheMode::JsonZip);
        assert_well_formed(n, prec, &nodes, &weights);

        // Only the .zip must exist after compute.
        assert!(
            zip_path.exists(),
            "fresh compute should write {} (the zip-only cache)",
            zip_path.display()
        );
        assert!(
            !json_path.exists(),
            "zip-only: fresh compute must not write an uncompressed .json"
        );

        // Round-trip: a second read goes through the zip again (no .json
        // exists) and must yield bit-identical nodes/weights.
        let (nodes2, weights2) = hp::gauss_legendre_nodes(n, prec, hp::CacheMode::JsonZip);
        assert_eq!(nodes.len(), nodes2.len());
        for (a, b) in nodes.iter().zip(nodes2.iter()) {
            assert_eq!(a.to_string(), b.to_string(), "node round-trip via zip");
        }
        for (a, b) in weights.iter().zip(weights2.iter()) {
            assert_eq!(a.to_string(), b.to_string(), "weight round-trip via zip");
        }
    }

    /// Sanity check that the legitimate computed GL nodes integrate a
    /// known polynomial correctly. This is a sanity wrapper around
    /// `gauss_legendre_compute` (no cache involved if the test runs
    /// in a fresh temp cwd).
    #[test]
    fn fresh_compute_integrates_x_squared() {
        let temp = fresh_temp_dir("integrate_x2");
        let _guard = CacheRootGuard::enter(&temp);

        // Use small n, modest precision: enough to validate, fast to run.
        let n = 16;
        let prec: u32 = 128;
        let (nodes, weights) = hp::gauss_legendre_nodes(n, prec, hp::CacheMode::JsonZip);
        assert_well_formed(n, prec, &nodes, &weights);

        // ∫_{-1}^{1} x² dx = 2/3.
        let mut sum = Float::with_val(prec, 0);
        for (x, w) in nodes.iter().zip(weights.iter()) {
            let mut term = x.clone();
            term.square_mut();
            term *= w;
            sum += &term;
        }
        let two_thirds = {
            let mut v = Float::with_val(prec, 2);
            v /= 3u32;
            v
        };
        let mut diff = sum.clone();
        diff -= &two_thirds;
        let abs_err = diff.abs();
        // GL-16 nails any polynomial of degree ≤ 31 to working precision.
        let tol = Float::with_val(prec, rug::Float::parse("1e-30").unwrap());
        assert!(
            abs_err.cmp_abs(&tol).map(|o| o.is_lt()).unwrap_or(false),
            "GL-{} integral of x² should match 2/3 to working precision; abs err = {}",
            n,
            abs_err
        );
    }

    /// HP GL nodes/weights satisfy classical structural identities:
    ///
    /// 1. **Symmetry**: nodes[i] = -nodes[n-1-i], weights[i] = weights[n-1-i]
    ///    on the symmetric interval [-1, 1].
    /// 2. **Sum of weights**: Σ w_i = 2 (= length of [-1, 1]).
    /// 3. **First moment**: Σ x_i · w_i = 0 (by symmetry of the
    ///    weight function on [-1, 1]).
    ///
    /// All three should hold to working precision regardless of `n`.
    #[test]
    fn hp_gl_nodes_satisfy_symmetry_and_moments() {
        let temp = fresh_temp_dir("symmetry_moments");
        let _guard = CacheRootGuard::enter(&temp);

        let n: usize = 12;
        let prec: u32 = 256;
        let (nodes, weights) = hp::gauss_legendre_nodes(n, prec, hp::CacheMode::JsonZip);
        assert_well_formed(n, prec, &nodes, &weights);

        // 1. Symmetry: nodes[i] + nodes[n-1-i] = 0.
        let tol = Float::with_val(prec, rug::Float::parse("1e-50").unwrap());
        for i in 0..n / 2 {
            let mut sum = nodes[i].clone();
            sum += &nodes[n - 1 - i];
            let abs_sum = sum.abs();
            assert!(
                abs_sum.cmp_abs(&tol).map(|o| o.is_lt()).unwrap_or(false),
                "node symmetry: nodes[{}] + nodes[{}] should be 0; got {}",
                i,
                n - 1 - i,
                abs_sum
            );
            // Weights mirror too.
            let mut wdiff = weights[i].clone();
            wdiff -= &weights[n - 1 - i];
            let abs_wdiff = wdiff.abs();
            assert!(
                abs_wdiff.cmp_abs(&tol).map(|o| o.is_lt()).unwrap_or(false),
                "weight symmetry: weights[{}] - weights[{}] should be 0; got {}",
                i,
                n - 1 - i,
                abs_wdiff
            );
        }

        // 2. Sum of weights = 2.
        let mut wsum = Float::with_val(prec, 0);
        for w in &weights {
            wsum += w;
        }
        let mut wdiff = wsum.clone();
        wdiff -= 2u32;
        let abs_wdiff = wdiff.abs();
        assert!(
            abs_wdiff.cmp_abs(&tol).map(|o| o.is_lt()).unwrap_or(false),
            "Σ weights should be 2; got {} (diff {})",
            wsum,
            abs_wdiff
        );

        // 3. First moment: Σ x_i · w_i = 0.
        let mut moment = Float::with_val(prec, 0);
        for (x, w) in nodes.iter().zip(weights.iter()) {
            let mut t = x.clone();
            t *= w;
            moment += &t;
        }
        let abs_moment = moment.abs();
        assert!(
            abs_moment.cmp_abs(&tol).map(|o| o.is_lt()).unwrap_or(false),
            "Σ x_i w_i should be 0; got {}",
            abs_moment
        );
    }

    /// A structurally-invalid `.json.zip` (shape OK, but values violate
    /// Σ w_i = 2 and antisymmetry) must be discarded by the
    /// structural validator, falling through to fresh compute.
    /// The bad file is preserved on disk (not deleted).
    #[test]
    fn cache_discards_structurally_invalid_json_and_recomputes() {
        let temp = fresh_temp_dir("structurally_invalid");
        let _guard = CacheRootGuard::enter(&temp);

        let n = 6;
        let prec: u32 = 128;
        let cache_dir = temp.join("data").join("gl_cache");
        std::fs::create_dir_all(&cache_dir).unwrap();

        // Plant a structurally-invalid envelope (all-zero nodes/weights:
        // Σw ≠ 2, nodes not antisymmetric) inside a .json.zip. The parser
        // accepts the envelope; the structural check rejects it, so the
        // loader falls through to fresh compute.
        let bad_payload = structurally_invalid_gl_json(n, prec);
        let zip_path = cache_dir.join(format!("prec{}_npts{}.json.zip", prec, n));
        {
            use std::io::Write;
            let f = std::fs::File::create(&zip_path).unwrap();
            let mut zw = zip::ZipWriter::new(f);
            let opts: zip::write::SimpleFileOptions = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated);
            zw.start_file(format!("prec{}_npts{}.json", prec, n), opts)
                .unwrap();
            zw.write_all(bad_payload.as_bytes()).unwrap();
            zw.finish().unwrap();
        }

        // Loading should silently fall through to fresh compute. The
        // returned values must be REAL GL nodes/weights (which pass the
        // structural check), not the bad payload's values.
        let (nodes, weights) = hp::gauss_legendre_nodes(n, prec, hp::CacheMode::JsonZip);
        assert_well_formed(n, prec, &nodes, &weights);

        // Sanity: real GL nodes are antisymmetric on [-1, 1] and
        // weights sum to 2. The bad payload had neither property.
        let mut wsum = Float::with_val(prec, 0);
        for w in &weights {
            wsum += w;
        }
        let mut wdiff = wsum.clone();
        wdiff -= 2u32;
        let abs_wdiff = wdiff.abs();
        let tol = Float::with_val(prec, rug::Float::parse("1e-30").unwrap());
        assert!(
            abs_wdiff.cmp_abs(&tol).map(|o| o.is_lt()).unwrap_or(false),
            "fresh compute should produce Σw=2; got Σw={}",
            wsum
        );

        // After fresh compute, save_gl_cache overwrites the bad .json.zip
        // with valid GL data (cleanup-then-write). Reading the zip now
        // must give real GL data that passes structural checks.
        assert!(
            zip_path.exists(),
            "the .json.zip should still exist after recompute"
        );
        let (nodes2, _w2) = hp::gauss_legendre_nodes(n, prec, hp::CacheMode::JsonZip);
        let mut antisym_ok = true;
        for (a, b) in nodes2.iter().zip(nodes2.iter().rev()) {
            let mut s = a.clone();
            s += b;
            if s.clone().abs() > Float::with_val(prec, rug::Float::parse("1e-30").unwrap()) {
                antisym_ok = false;
                break;
            }
        }
        assert!(
            antisym_ok,
            "after recompute the cached zip should hold valid antisymmetric GL nodes"
        );
    }

    /// A truncated/corrupt `.json.zip` must be detected and skipped
    /// without panic, with the cache falling through to fresh compute.
    #[test]
    fn cache_handles_corrupt_zip_gracefully() {
        let temp = fresh_temp_dir("corrupt_zip");
        let _guard = CacheRootGuard::enter(&temp);

        let n = 4;
        let prec: u32 = 64;
        let cache_dir = temp.join("data").join("gl_cache");
        std::fs::create_dir_all(&cache_dir).unwrap();

        // Write a "zip" that's actually random garbage. zip::ZipArchive::new
        // should reject it; our loader handles that and falls through.
        let zip_path = cache_dir.join(format!("prec{}_npts{}.json.zip", prec, n));
        std::fs::write(&zip_path, b"not a zip file at all -- random bytes").unwrap();

        // Should not panic. Should fall through to fresh compute.
        let (nodes, weights) = hp::gauss_legendre_nodes(n, prec, hp::CacheMode::JsonZip);
        assert_well_formed(n, prec, &nodes, &weights);

        // The corrupt file is preserved on disk.
        assert!(
            zip_path.exists(),
            "corrupt zip should be preserved on disk for the user to inspect"
        );
    }

    /// `verify_gl_cache_dir` reports OK for valid cache files and
    /// `StructurallyInvalid` for corrupt ones, without modifying any files.
    #[test]
    fn verify_gl_cache_dir_reports_per_file_status() {
        use hp::CacheFileStatus;

        let temp = fresh_temp_dir("verify_dir");
        let _guard = CacheRootGuard::enter(&temp);

        let cache_dir = temp.join("data").join("gl_cache");
        std::fs::create_dir_all(&cache_dir).unwrap();

        // 1. Valid file: well-formed envelope + structurally valid values.
        // Use real GL-4 nodes/weights from a fresh compute so structural
        // checks pass.
        let (real_nodes, real_weights) = hp::gauss_legendre_nodes(4, 64, hp::CacheMode::Off);
        let ns: Vec<String> = real_nodes.iter().map(|f| f.to_string()).collect();
        let ws: Vec<String> = real_weights.iter().map(|f| f.to_string()).collect();
        let valid_json = serde_json::json!({
            "schema_version": 1,
            "toolkit_version": hp::toolkit_version_for_test(),
            "n_pts": 4_usize,
            "precision_bits": 64_u32,
            "nodes": ns,
            "weights": ws,
        })
        .to_string();
        let valid_path = cache_dir.join("prec64_npts4.json");
        std::fs::write(&valid_path, valid_json).unwrap();

        // 2. Structurally-invalid file: valid envelope but nodes/weights
        //    are all zeros → fails Σw=2 identity.
        let bad_ns: Vec<String> = (0..5).map(|_| "0".to_string()).collect();
        let bad_ws: Vec<String> = (0..5).map(|_| "0".to_string()).collect();
        let bad_json = serde_json::json!({
            "schema_version": 1,
            "toolkit_version": hp::toolkit_version_for_test(),
            "n_pts": 5_usize,
            "precision_bits": 64_u32,
            "nodes": bad_ns,
            "weights": bad_ws,
        })
        .to_string();
        let bad_path = cache_dir.join("prec64_npts5.json");
        std::fs::write(&bad_path, bad_json).unwrap();

        // 3. Unrecognized filename — should be reported as Skipped.
        let skipped_path = cache_dir.join("not_a_cache_file.txt");
        std::fs::write(&skipped_path, "irrelevant").unwrap();

        // 4. File matching the pattern but malformed JSON.
        let malformed_path = cache_dir.join("prec64_npts3.json");
        std::fs::write(&malformed_path, "{").unwrap();

        let report = hp::verify_gl_cache_dir(&cache_dir).unwrap();
        assert_eq!(report.directory, cache_dir);
        // 4 entries; one OK, one StructurallyInvalid, one Skipped, one LoadFailed.
        assert_eq!(
            report.statuses.len(),
            4,
            "expected 4 statuses (one per file); got {}",
            report.statuses.len()
        );

        let mut saw_ok = false;
        let mut saw_invalid = false;
        let mut saw_skipped = false;
        let mut saw_loadfail = false;
        for s in &report.statuses {
            match s {
                CacheFileStatus::Ok { path, n, prec } => {
                    assert_eq!(path, &valid_path);
                    assert_eq!(*n, 4);
                    assert_eq!(*prec, 64);
                    saw_ok = true;
                }
                CacheFileStatus::StructurallyInvalid { path, n, prec, .. } => {
                    assert_eq!(path, &bad_path);
                    assert_eq!(*n, 5);
                    assert_eq!(*prec, 64);
                    saw_invalid = true;
                }
                CacheFileStatus::Skipped { path, .. } => {
                    assert_eq!(path, &skipped_path);
                    saw_skipped = true;
                }
                CacheFileStatus::LoadFailed { path, n, prec, .. } => {
                    assert_eq!(path, &malformed_path);
                    assert_eq!(*n, 3);
                    assert_eq!(*prec, 64);
                    saw_loadfail = true;
                }
            }
        }
        assert!(saw_ok, "missing Ok status");
        assert!(saw_invalid, "missing StructurallyInvalid status");
        assert!(saw_skipped, "missing Skipped status");
        assert!(saw_loadfail, "missing LoadFailed status");

        assert_eq!(report.ok_count(), 1);
        assert_eq!(
            report.failure_count(),
            2,
            "LoadFailed + StructurallyInvalid both count as failures; expected 2"
        );

        // Files preserved (verify_gl_cache_dir is read-only).
        assert!(valid_path.exists());
        assert!(bad_path.exists());
        assert!(skipped_path.exists());
        assert!(malformed_path.exists());
    }

    /// `verify_gl_cache_dir` on a non-existent directory returns an
    /// empty report, not an error.
    #[test]
    fn verify_gl_cache_dir_handles_missing_directory() {
        let temp = fresh_temp_dir("verify_missing");
        let _guard = CacheRootGuard::enter(&temp);

        let nonexistent = temp.join("does_not_exist");
        let report = hp::verify_gl_cache_dir(&nonexistent).unwrap();
        assert_eq!(report.statuses.len(), 0);
        assert_eq!(report.ok_count(), 0);
        assert_eq!(report.failure_count(), 0);
    }
}

#[cfg(all(test, feature = "hp"))]
mod cache_envelope_regression {
    use super::hp;

    #[test]
    fn mislabeled_cache_envelopes_are_rejected_before_numeric_decode() {
        let good = serde_json::json!({
            "schema_version": 1, "toolkit_version": hp::toolkit_version_for_test(),
            "n_pts": 2, "precision_bits": 128,
            "nodes": ["-0.5", "0.5"], "weights": ["1", "1"]
        });
        assert!(hp::parse_gl_json_for_test(&good.to_string(), 2, 128).is_some());
        for (field, value) in [("schema_version", 2), ("n_pts", 3), ("precision_bits", 64)] {
            let mut bad = good.clone();
            bad[field] = serde_json::json!(value);
            assert!(hp::parse_gl_json_for_test(&bad.to_string(), 2, 128).is_none());
        }
    }
}
