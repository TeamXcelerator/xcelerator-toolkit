// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Gauss-Legendre quadrature at f64 and high precision.
//!
//! The HP nodes/weights are computed via Newton iteration on Legendre
//! polynomials and cached to disk under `$XC_CACHE_ROOT/gl_cache/`, or under
//! the per-user cache root when that variable is unset, so they're reused
//! across runs at the same `(n_pts, precision_bits)`.
//!
//! The standalone HP API reads a local deterministic `.json.zip`, directly
//! in memory, and computes and stores that representation on a miss.
//! Remote local/private/public resolution belongs exclusively to the managed
//! cache fabric exposed by `gauss_legendre_nodes_via_cache`.
//!
//! `CacheMode` selects local cache behavior:
//! - `Off`          — no cache read or write; always compute.
//! - `JsonOnly`     — deprecated compatibility name; computes without reads or writes.
//! - `JsonZip`      — local compressed cache (**default**).
//!
//! `gauss_legendre_nodes(n, prec, mode)` takes the `CacheMode`
//! explicitly; pass `CacheMode::default()` for standard local behavior.
//!
//! New computes write only the compressed representation. Every reused HP rule
//! passes the full O(n^2) Legendre/node/weight check, once per process for each
//! exact managed-cache payload. Legacy standalone files
//! establish numerical compatibility, not bit-for-bit producer authenticity:
//! accepted values may differ from a fresh rule within the explicit validation
//! tolerances. Managed identities bind generation semantics, but are not by
//! themselves proof that an untrusted producer executed that algorithm.
//! HP rule accuracy is absolute at the working scale, not uniformly relative
//! in small endpoint weights. Their relative sensitivity grows with order; an
//! odd central node need not be exactly zero. Orders unresolved at the selected
//! precision fail validation rather than imply a certified rule.

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

#[path = "binary64_triple_sum.rs"]
mod binary64_triple_sum;
/// Accumulate the exact products of stored binary64 weights, samples and
/// half-width, then round their sum once to nearest-even binary64.
pub const GAUSS_LEGENDRE_ACCUMULATION_SEMANTICS: &str = "gauss-legendre-exact-stored-triple-sum-v2";

/// Approximates an integral with an `n`-point Gauss--Legendre rule.
///
/// # Mathematical semantics
/// Maps the canonical nodes on `[-1, 1]` to `[a, b]` and returns the weighted
/// sum. It is exact for polynomials through degree `2n - 1` in exact
/// arithmetic; it does not prove an error bound for a general integrand.
///
/// # Precision
/// Nodes, weights, and function values use binary64. Products and accumulation
/// are exact for those stored values, with one final nearest-even rounding. Choose
/// an HP quadrature route when binary64 rounding is not an explicit policy.
///
/// # Failure states
/// This convenience function returns NaN for invalid order, nonfinite bounds
/// or samples, and unrepresentable arithmetic. Use
/// [`try_gauss_legendre_npt_f64`] for an explicit error. Positive order is required.
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
    try_gauss_legendre_npt_f64(f, a, b, n).unwrap_or(f64::NAN)
}

/// Checked computed quadrature. Reversed finite bounds give an oriented
/// integral; equal finite bounds return zero. Invalid order, nonfinite samples,
/// or unrepresentable intermediate arithmetic are explicit errors. This does
/// not bound discretization error or callback error.
pub fn try_gauss_legendre_npt_f64<F: Fn(f64) -> f64>(
    f: F,
    a: f64,
    b: f64,
    n: usize,
) -> anyhow::Result<f64> {
    anyhow::ensure!(
        a.is_finite() && b.is_finite(),
        "quadrature bounds must be finite"
    );
    let (nodes, weights) = try_gl_nodes_weights_f64(n)?;
    if a == b {
        return Ok(0.0);
    }
    // Keep ordinary arithmetic unchanged. Half-sums/differences handle cases
    // where only the unscaled sum/difference exceeds binary64's range.
    let sum = a + b;
    let difference = b - a;
    let mid = if sum.is_finite() {
        0.5 * sum
    } else {
        0.5 * a + 0.5 * b
    };
    let half = if difference.is_finite() {
        0.5 * difference
    } else {
        0.5 * b - 0.5 * a
    };
    anyhow::ensure!(half != 0.0, "quadrature half-width is unrepresentable");
    let mut sum = binary64_triple_sum::TripleSum::new();
    for (node, weight) in nodes.into_iter().zip(weights) {
        let x = mid + half * node;
        anyhow::ensure!(
            x.is_finite() && x >= a.min(b) && x <= a.max(b),
            "quadrature node is unrepresentable or outside the interval"
        );
        let sample = f(x);
        anyhow::ensure!(sample.is_finite(), "quadrature sample must be finite");
        sum.add([weight, sample, half])?;
    }
    sum.finish()
}

/// Compute n-point Gauss-Legendre nodes and weights on `[-1, 1]` at f64.
/// Accuracy is absolute at the working scale; small endpoint weights lose
/// relative accuracy as order grows. This computed rule is not a certified
/// uniform relative-error enclosure.
///
/// The nodes are in descending order, with weights in the same order. Callers that need to set up
/// a variable-integrand integral (e.g. a complex-valued integrand with
/// multiple accumulator sums) can call this directly rather than using
/// [`gauss_legendre_npt_f64`], which takes a single closure.
/// Panics for invalid order or a nonrepresentable rule; use
/// [`try_gl_nodes_weights_f64`] for an explicit error.
pub fn gl_nodes_weights_f64(n: usize) -> (Vec<f64>, Vec<f64>) {
    try_gl_nodes_weights_f64(n).expect("valid representable Gauss-Legendre rule required")
}

/// Checked native node/weight construction for positive order. Newton roots
/// remain computed approximations, not certified root enclosures.
pub fn try_gl_nodes_weights_f64(n: usize) -> anyhow::Result<(Vec<f64>, Vec<f64>)> {
    let denominator = n.checked_mul(4).and_then(|value| value.checked_add(2));
    anyhow::ensure!(
        n > 0 && denominator.is_some_and(|value| (value as u64) <= (1_u64 << 53)),
        "quadrature order must be positive with exactly representable recurrence indices"
    );
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
    anyhow::ensure!(
        nodes.iter().all(|x| x.is_finite() && x.abs() < 1.0)
            && weights.iter().all(|w| w.is_finite() && *w > 0.0)
            && nodes.windows(2).all(|pair| pair[0] > pair[1]),
        "Gauss-Legendre iteration produced an invalid rule"
    );
    Ok((nodes, weights))
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
        /// Present only when this rule was reused or written to the cache.
        /// Disabled and nonwriting cache misses return no persisted identity.
        pub artifact_manifest: Option<xc_cache::ArtifactManifest>,
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

    fn validate_gl_domain(n: usize, prec: u32) -> Result<(), CacheError> {
        if n == 0
            || !(16..=1_000_000).contains(&prec)
            || n.checked_mul(4)
                .and_then(|n| n.checked_add(2))
                .and_then(|n| i64::try_from(n).ok())
                .is_none()
        {
            return Err(CacheError::InvalidManifest(
                "GL requires positive representable order and precision 16..=1000000".into(),
            ));
        }
        Ok(())
    }

    fn full_cache_check(nodes: &[Float], weights: &[Float], prec: u32) -> Option<String> {
        match check_gauss_legendre_rule_hp(nodes, weights, prec, nodes.len()) {
            Ok(check) if check.checks_passed => None,
            Ok(_) => Some("Gauss-Legendre root or derivative-weight identity mismatch".into()),
            Err(error) => Some(error.to_string()),
        }
    }

    /// Rules that already passed the full O(n^2) check in this process, keyed by
    /// their exact payload. The check is a pure function of these bytes, so a
    /// later lookup of the same rule returns the checked values instead of
    /// repeating it. Bounded to a few rules.
    static VALIDATED_TABLES: std::sync::Mutex<Vec<(PortableGlTable, GlTable)>> =
        std::sync::Mutex::new(Vec::new());
    const VALIDATED_TABLE_LIMIT: usize = 8;

    fn decode_portable_table(
        table: &PortableGlTable,
        n: usize,
        prec: u32,
    ) -> Result<GlTable, CacheError> {
        validate_gl_domain(n, prec)?;
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
        if let Some((_, checked)) = VALIDATED_TABLES
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .find(|(validated, _)| validated == table)
        {
            return Ok(checked.clone());
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
        if let Some(reason) = full_cache_check(&nodes, &weights, prec) {
            return Err(CacheError::InvalidManifest(format!(
                "quadrature payload failed structural validation: {reason}"
            )));
        }
        let mut validated = VALIDATED_TABLES
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if validated.len() >= VALIDATED_TABLE_LIMIT {
            validated.remove(0);
        }
        validated.push((table.clone(), (nodes.clone(), weights.clone())));
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
    /// Reused and fresh rules must pass structural moments and every Legendre
    /// root/derivative-weight identity before return. Validation costs O(n^2)
    /// point arithmetic and does not establish an interval certificate.
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
        validate_gl_domain(n, precision_bits)?;
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
            minimum_reader_version: ToolkitVersion::parse(xc_cache::CLEAN_SLATE)?,
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
                let started = std::time::Instant::now();
                xc_core::progress_message!(
                    "[HP] Gauss-Legendre rule n={n}, {precision_bits} bits: computing {} roots",
                    if root_schedule.parallel_min_task_len().is_some() {
                        "parallel"
                    } else {
                        "serial"
                    }
                );
                let table = gauss_legendre_compute_scheduled(n, precision_bits, root_schedule)?;
                xc_core::progress_message!(
                    "[HP] Gauss-Legendre rule n={n}, {precision_bits} bits: computed in {:.1}s",
                    started.elapsed().as_secs_f64()
                );
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
        let artifact_manifest = resolved.produced_manifest.or(resolved.reused_manifest);
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

    /// Screen finite interior ordered nodes, positive weights, symmetry, and
    /// even moments through degree six, including the classical Gauss-
    /// Legendre node/weight pair on `[-1, 1]`:
    ///
    ///   1. Σ w_i = 2 (length of [-1, 1])
    ///   2. Σ x_i · w_i = 0 (first moment of an even weight function)
    ///   3. Antisymmetry: nodes[i] + nodes[n-1-i] = 0,
    ///      weights[i] - weights[n-1-i] = 0
    ///
    /// Returns `None` if every structural identity holds within
    /// `cache_structural_tol(prec)`. Returns `Some(reason)` describing
    /// the first identity that fails, with both magnitudes in the
    /// reason string for diagnostic purposes.
    ///
    /// Used by `load_gl_cache` to discard structurally-broken cache
    /// files (e.g. wrong precision, value corruption, accidental edit)
    /// before they pollute downstream HP integration.
    pub(super) fn cache_structural_check(
        nodes: &[Float],
        weights: &[Float],
        prec: u32,
    ) -> Option<String> {
        let n = nodes.len();
        if weights.len() != n {
            return Some(format!(
                "weight count {} != node count {}",
                weights.len(),
                n
            ));
        }
        if n == 0 {
            return Some("empty Gauss-Legendre rule".to_owned());
        }
        for (i, (x, w)) in nodes.iter().zip(weights).enumerate() {
            if !x.is_finite() || !w.is_finite() || x <= &-1 || x >= &1 || w <= &0 {
                return Some(format!(
                    "invalid interior node or positive finite weight at {i}"
                ));
            }
            if i > 0 && nodes[i - 1] >= *x {
                return Some(format!("nodes are not strictly increasing at {i}"));
            }
        }
        let tol = cache_structural_tol(prec);
        // Cheap O(n) independent even moments. This detects many symmetric
        // corruptions that mass and the first moment alone cannot detect.
        for degree in [2u32, 4, 6] {
            if degree as usize >= 2 * n {
                continue;
            }
            let mut moment = Float::with_val(prec, 0);
            for (x, w) in nodes.iter().zip(weights) {
                let mut term = Float::with_val(prec, 1);
                for _ in 0..degree {
                    term *= x;
                }
                term *= w;
                moment += term;
            }
            let mut expected = Float::with_val(prec, 2);
            expected /= degree + 1;
            moment -= expected;
            if moment.abs() >= tol {
                return Some(format!("degree-{degree} Gauss-Legendre moment mismatch"));
            }
        }

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
    /// # Panics
    /// Panics on invalid order/precision or failed generation/validation.
    /// Use the fallible managed API when errors must be returned.
    /// Nodes are strictly ascending, with corresponding positive weights.
    pub fn gauss_legendre_nodes(n: usize, prec: u32, mode: CacheMode) -> (Vec<Float>, Vec<Float>) {
        gauss_legendre_nodes_impl(
            n,
            prec,
            mode,
            crate::hp_runtime::GlRootSchedule::serial(),
            true,
        )
        .expect("valid finite HP Gauss-Legendre rule required")
    }

    /// Schedule-aware compatibility-cache entry point used by the CCM batch
    /// planner. The scheduling decision has already been made by the owner.
    #[doc(hidden)]
    /// # Panics
    /// Panics on invalid order/precision or failed generation/validation.
    pub fn gauss_legendre_nodes_scheduled(
        n: usize,
        prec: u32,
        mode: CacheMode,
        root_schedule: crate::hp_runtime::GlRootSchedule,
    ) -> (Vec<Float>, Vec<Float>) {
        gauss_legendre_nodes_impl(n, prec, mode, root_schedule, false)
            .expect("valid finite HP Gauss-Legendre rule required")
    }

    /// Checked scheduled resolver. Numerical failures propagate to the owning batch.
    #[doc(hidden)]
    pub fn try_gauss_legendre_nodes_scheduled(
        n: usize,
        prec: u32,
        mode: CacheMode,
        root_schedule: crate::hp_runtime::GlRootSchedule,
    ) -> Result<GlTable, CacheError> {
        gauss_legendre_nodes_impl(n, prec, mode, root_schedule, false)
    }

    /// Checked compatibility-cache resolver; root/weight identities are checked
    /// on both fresh and reused rules. These are point checks, not certificates.
    pub fn try_gauss_legendre_nodes(
        n: usize,
        prec: u32,
        mode: CacheMode,
    ) -> Result<GlTable, CacheError> {
        gauss_legendre_nodes_impl(
            n,
            prec,
            mode,
            crate::hp_runtime::GlRootSchedule::serial(),
            true,
        )
    }

    fn gauss_legendre_nodes_impl(
        n: usize,
        prec: u32,
        mode: CacheMode,
        root_schedule: crate::hp_runtime::GlRootSchedule,
        top_level: bool,
    ) -> Result<GlTable, CacheError> {
        validate_gl_domain(n, prec)?;
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
                return Ok(cached);
            }
        }
        performance_resolve.set_cache_disposition("computed");
        let performance_construct =
            xc_core::performance_stage_with("quadrature.gl.construct", || {
                gl_performance_metadata_scheduled(n, prec, root_schedule)
            });
        let result = gauss_legendre_compute_scheduled(n, prec, root_schedule)?;
        drop(performance_construct);
        if mode != CacheMode::Off {
            let performance_store =
                xc_core::performance_stage_with("quadrature.gl.compatibility_store", || {
                    gl_performance_metadata(n, prec)
                });
            save_gl_cache(n, prec, &result.0, &result.1, mode);
            drop(performance_store);
        }
        Ok(result)
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
    /// `gl_cache` under the per-user cache root shared with the managed
    /// cache ([`xc_core::default_cache_root`]). Created on demand. The
    /// working directory is never used, so runs started inside a checkout
    /// do not write into it.
    fn gl_cache_dir() -> Option<std::path::PathBuf> {
        #[cfg(test)]
        if let Some(root) = TEST_CACHE_ROOT.with(|current| current.borrow().clone()) {
            let dir = root.join("gl_cache");
            std::fs::create_dir_all(&dir).ok()?;
            return Some(dir);
        }
        let dir = xc_core::configured_cache_root().join("gl_cache");
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
    #[cfg(test)]
    fn parse_gl_json(data: &str, n: usize, prec: u32) -> Option<GlTable> {
        let table = parse_gl_envelope_unchecked(data, n, prec)?;
        if full_cache_check(&table.0, &table.1, prec).is_some() {
            return None;
        }
        Some(table)
    }

    // Untrusted decoded values for the verifier's error classification. A
    // production result must pass full_cache_check before it can be returned.
    fn parse_gl_envelope_unchecked(data: &str, n: usize, prec: u32) -> Option<GlTable> {
        validate_gl_domain(n, prec).ok()?;
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
        xc_core::progress_message!(
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
            Ok((table, _)) => Some(table),
            Err(error) => {
                warn_cache_skip(&zip_path, &error.reason());
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

    #[derive(Debug)]
    enum GlCacheReadError {
        Load(String),
        Stale { found: String, minimum: String },
        Numerical(String),
    }
    impl GlCacheReadError {
        fn reason(&self) -> String {
            match self {
                Self::Load(reason) => reason.clone(),
                Self::Stale { found, minimum } => {
                    format!("stale toolkit version {found}; minimum {minimum}")
                }
                Self::Numerical(reason) => format!("numerical rule validation failed: {reason}"),
            }
        }
    }
    // Runtime and verifier use exactly the same container, version, envelope,
    // and full numerical acceptance path.
    fn load_from_zip(
        zip_path: &std::path::Path,
        n: usize,
        prec: u32,
    ) -> Result<(GlTable, String), GlCacheReadError> {
        let data = read_gl_zip_payload(zip_path, n, prec).ok_or_else(|| {
            GlCacheReadError::Load(
                "ZIP container, entry identity, decoding, or decoded-size check failed".to_owned(),
            )
        })?;
        let value: serde_json::Value = serde_json::from_str(&data)
            .map_err(|e| GlCacheReadError::Load(format!("JSON parse failed: {e}")))?;
        let version = value
            .get("toolkit_version")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                GlCacheReadError::Load("toolkit_version is absent or not a string".to_owned())
            })?;
        let minimum = effective_min_version();
        if version_is_older(version, &minimum) {
            return Err(GlCacheReadError::Stale {
                found: version.to_owned(),
                minimum,
            });
        }
        let table = parse_gl_envelope_unchecked(&data, n, prec).ok_or_else(|| {
            GlCacheReadError::Load(
                "GL envelope identity, shape, domain, or decimal parse failed".to_owned(),
            )
        })?;
        if let Some(reason) = full_cache_check(&table.0, &table.1, prec) {
            return Err(GlCacheReadError::Numerical(reason));
        }
        Ok((table, data))
    }

    fn read_gl_zip_payload(zip_path: &std::path::Path, n: usize, prec: u32) -> Option<String> {
        use std::io::Read;
        let maximum_bytes = (n as u64)
            .checked_mul(2)?
            .checked_mul(u64::from(prec).checked_add(128)?)?
            .checked_add(4096)?;
        let file = std::fs::File::open(zip_path).ok()?;
        if file.metadata().ok()?.len() > maximum_bytes.checked_add(65536)? {
            return None;
        }
        let mut archive = zip::ZipArchive::new(file).ok()?;
        if archive.len() != 1 {
            return None;
        }
        let entry_name = format!("prec{}_npts{}.json", prec, n);
        let mut entry = archive.by_name(&entry_name).ok()?;
        // A scalar needs fewer than `prec` decimal characters plus sign and
        // exponent overhead. Bound both the ZIP header and actual expansion;
        // do not trust a compressed file to dictate an unbounded allocation.

        if entry.size() > maximum_bytes {
            return None;
        }
        let mut data = String::new();
        entry
            .by_ref()
            .take(maximum_bytes.checked_add(1)?)
            .read_to_string(&mut data)
            .ok()?;
        if data.len() as u64 > maximum_bytes {
            return None;
        }
        Some(data)
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
            if let Err(error) = xc_cache::atomic_replace_cache_file(&zip_path, &buf) {
                xc_core::progress_message!("quadrature cache write failed: {error}");
            }
        }
    }

    #[cfg(test)]
    pub(super) fn gauss_legendre_compute(n: usize, prec: u32) -> (Vec<Float>, Vec<Float>) {
        gauss_legendre_compute_scheduled(n, prec, crate::hp_runtime::GlRootSchedule::serial())
            .expect("valid test GL rule")
    }

    fn gauss_legendre_compute_scheduled(
        n: usize,
        prec: u32,
        root_schedule: crate::hp_runtime::GlRootSchedule,
    ) -> Result<GlTable, CacheError> {
        validate_gl_domain(n, prec)?;
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
        if combined
            .iter()
            .any(|(x, w)| !x.is_finite() || !w.is_finite())
        {
            return Err(CacheError::InvalidManifest(
                "GL Newton arithmetic is nonfinite".into(),
            ));
        }
        combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let result: GlTable = combined.into_iter().unzip();
        if let Some(reason) = full_cache_check(&result.0, &result.1, prec) {
            return Err(CacheError::InvalidManifest(format!(
                "fresh GL rule failed validation: {reason}"
            )));
        }
        Ok(result)
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
    #[test]
    fn validated_rules_are_remembered_and_altered_rules_still_rejected() {
        let (n, prec) = (12, 192);
        let rule = portable_table(
            n,
            prec,
            gauss_legendre_compute_scheduled(n, prec, crate::hp_runtime::GlRootSchedule::serial())
                .unwrap(),
        );
        let first = decode_portable_table(&rule, n, prec).unwrap();
        assert!(VALIDATED_TABLES
            .lock()
            .unwrap()
            .iter()
            .any(|(table, _)| table == &rule));
        let second = decode_portable_table(&rule, n, prec).unwrap();
        assert_eq!(first, second);
        let mut altered = rule.clone();
        altered.nodes.swap(0, 1);
        assert!(decode_portable_table(&altered, n, prec).is_err());
        let mut altered = rule;
        altered.weights[0] = "0".into();
        assert!(decode_portable_table(&altered, n, prec).is_err());
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

    /// Computed full-rule diagnostics, also required at cache/read boundaries.
    #[derive(Clone, Debug)]
    pub struct GaussLegendreRuleCheckHp {
        pub order: usize,
        pub precision_bits: u32,
        /// Largest absolute Legendre polynomial residual (legacy diagnostic).
        pub maximum_legendre_residual: Float,
        /// Largest relative derivative-weight defect (legacy diagnostic).
        pub maximum_relative_weight_defect: Float,
        /// Largest Newton correction |P_n(x)/P_n'(x)|, the node's distance
        /// to its Legendre root to first order.
        pub maximum_node_correction: Float,
        /// Largest |w - 2/((1-x^2) P_n'(x)^2)|.
        pub maximum_weight_error: Float,
        pub tolerance: Float,
        pub checks_passed: bool,
    }

    /// Check every Legendre root and derivative-weight identity in O(n^2)
    /// arithmetic. The explicit order budget is enforced before recurrence
    /// work. The Newton-correction estimate and absolute derivative-weight
    /// defect must each be at most 16 * 2^-prec. These are point-arithmetic
    /// diagnostics, not a root-distance bound or a correct-rounding certificate.
    pub fn check_gauss_legendre_rule_hp(
        nodes: &[Float],
        weights: &[Float],
        prec: u32,
        maximum_order: usize,
    ) -> anyhow::Result<GaussLegendreRuleCheckHp> {
        if !(16..=1_000_000).contains(&prec)
            || nodes.len() > maximum_order
            || nodes.is_empty()
            || nodes.iter().chain(weights).any(|x| x.prec() > prec)
        {
            anyhow::bail!(
                "invalid GL precision/order budget or input precision exceeds declared precision"
            );
        }
        if let Some(reason) = cache_structural_check(nodes, weights, prec) {
            anyhow::bail!(reason);
        }
        let n = nodes.len();
        validate_gl_domain(n, prec)?;
        let guard = prec + 64;
        let mut root_max = Float::with_val(guard, 0);
        let mut relative_weight_max = Float::with_val(guard, 0);
        let mut correction_max = Float::with_val(guard, 0);
        let mut weight_max = Float::with_val(guard, 0);
        for (source_x, source_w) in nodes.iter().zip(weights) {
            let x = Float::with_val(guard, source_x);
            let (pn, dpn) = legendre_p_and_deriv(n, &x, guard);
            if !pn.is_finite() || !dpn.is_finite() || dpn.is_zero() {
                anyhow::bail!("nonfinite GL recurrence");
            }
            // A per-node scale: |P_n'| ranges from about sqrt(n) inside the
            // interval to n^2/2 at its ends, so |P_n| alone is not comparable.
            root_max = root_max.max(&pn.clone().abs());
            let correction = Float::with_val(guard, &pn / &dpn).abs();
            let mut formula = Float::with_val(guard, 1);
            formula -= x.clone().square();
            formula *= dpn.square();
            let mut relative_defect = Float::with_val(guard, &formula * source_w);
            relative_defect /= 2;
            relative_defect -= 1;
            relative_weight_max = relative_weight_max.max(&relative_defect.abs());
            let formula = Float::with_val(guard, 2) / formula;
            let error = Float::with_val(guard, source_w - &formula).abs();
            if !correction.is_finite() || !error.is_finite() {
                anyhow::bail!("nonfinite GL weight check");
            }
            if correction > correction_max {
                correction_max = correction;
            }
            if error > weight_max {
                weight_max = error;
            }
        }
        // Correctly rounded tables measure below 2 * 2^-prec on both counts.
        let tolerance = Float::with_val(guard, 16) >> prec;
        Ok(GaussLegendreRuleCheckHp {
            order: n,
            precision_bits: prec,
            checks_passed: correction_max <= tolerance && weight_max <= tolerance,
            maximum_legendre_residual: root_max,
            maximum_relative_weight_defect: relative_weight_max,
            maximum_node_correction: correction_max,
            maximum_weight_error: weight_max,
            tolerance,
        })
    }

    // ===========================================================================
    // Public cache-verification API
    // ===========================================================================

    pub const GL_CACHE_ADMISSION_SEMANTICS: &str =
        "gl-zip-runtime-aligned-version-envelope-full-rule-admission-v2";

    /// Per-file outcome from `verify_gl_cache_dir`.
    #[derive(Debug, Clone)]
    pub enum CacheFileStatus {
        /// File loaded and passed structural and full Legendre node/weight checks.
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
        /// A valid version string is older than the active producer floor.
        Stale {
            path: std::path::PathBuf,
            n: usize,
            prec: u32,
            found_version: String,
            minimum_version: String,
        },
        /// File loaded successfully but failed at least one of the GL
        /// structural identities or the full Legendre node/weight validation.
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
                        CacheFileStatus::Stale { .. }
                            | CacheFileStatus::LoadFailed { .. }
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
                    CacheFileStatus::Stale { .. }
                        | CacheFileStatus::LoadFailed { .. }
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

    /// Walk the directory and apply runtime admission to every
    /// `prec{P}_npts{N}.json.zip` file. Plain JSON is skipped because the runtime
    /// does not read it. Returns a per-file
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

            if !name.ends_with(".json.zip") {
                statuses.push(CacheFileStatus::Skipped {
                    path,
                    reason: "plain JSON is not read by the runtime ZIP-only cache".to_owned(),
                });
                continue;
            }
            statuses.push(match load_from_zip(&path, n, prec) {
                Ok(_) => CacheFileStatus::Ok { path, n, prec },
                Err(GlCacheReadError::Stale { found, minimum }) => CacheFileStatus::Stale {
                    path,
                    n,
                    prec,
                    found_version: found,
                    minimum_version: minimum,
                },
                Err(GlCacheReadError::Load(reason)) => CacheFileStatus::LoadFailed {
                    path,
                    n,
                    prec,
                    reason,
                },
                Err(GlCacheReadError::Numerical(reason)) => CacheFileStatus::StructurallyInvalid {
                    path,
                    n,
                    prec,
                    reason,
                },
            });
        }

        Ok(CacheVerifyReport {
            directory: dir.to_path_buf(),
            statuses,
        })
    }
}

#[cfg(feature = "hp")]
pub use hp::{
    check_gauss_legendre_rule_hp, gauss_legendre_nodes, gauss_legendre_nodes_scheduled,
    gauss_legendre_nodes_via_cache, gauss_legendre_nodes_via_cache_scheduled,
    try_gauss_legendre_nodes, try_gauss_legendre_nodes_scheduled, verify_gl_cache_dir,
    CacheFileStatus, CacheMode, CacheVerifyReport, CachedQuadratureRule, GaussLegendreRuleCheckHp,
    QuadratureCacheRequest, GL_CACHE_ADMISSION_SEMANTICS,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "hp")]
    #[test]
    fn hp_cache_rejects_non_gaussian_rule_with_matching_low_moments() {
        use rug::Float;
        let p = 256;
        let a = Float::with_val(p, rug::Rational::from((3, 4))).sqrt();
        let b = Float::with_val(p, rug::Rational::from((1, 7))).sqrt();
        let wa = Float::with_val(p, rug::Rational::from((256, 765)));
        let wb = Float::with_val(p, rug::Rational::from((49, 85)));
        let nodes = vec![-a.clone(), -b.clone(), Float::with_val(p, 0), b, a];
        let weights = vec![
            wa.clone(),
            wb.clone(),
            Float::with_val(p, rug::Rational::from((8, 45))),
            wb,
            wa,
        ];
        // This positive symmetric five-node rule integrates degrees 0..=7
        // exactly, but not degree 8: it is not the five-node Gaussian rule.
        assert!(
            hp::cache_structural_check(&nodes, &weights, p).is_none(),
            "counterexample must pass the old cheap screen"
        );
        let payload = serde_json::json!({"schema_version":1,"toolkit_version":env!("CARGO_PKG_VERSION"),"n_pts":5,"precision_bits":p,"nodes":nodes.iter().map(Float::to_string).collect::<Vec<_>>(),"weights":weights.iter().map(Float::to_string).collect::<Vec<_>>()});
        assert!(
            hp::parse_gl_json_for_test(&payload.to_string(), 5, p).is_none(),
            "cache accepted a non-Gaussian rule"
        );
    }

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

        let root = xc_core::test_support::TestDir::new("quad-fabric");
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
            current_toolkit_version: ToolkitVersion::parse("0.16.0").unwrap(),
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

    /// Make a fresh, unique throwaway directory outside the checkout. The
    /// returned guard deletes the directory when dropped, including on
    /// panic, so callers must keep it alive for the whole test and declare
    /// it before any `CacheRootGuard` that points into it.
    fn fresh_temp_dir(tag: &str) -> xc_core::test_support::TestDir {
        xc_core::test_support::TestDir::new(tag)
    }

    #[test]
    fn rule_validation_rejects_symmetric_but_wrong_rules() {
        let p = 128;
        let (nodes, weights) = hp::gauss_legendre_nodes(8, p, hp::CacheMode::Off);
        assert!(hp::cache_structural_check(&nodes, &weights, p).is_none());
        assert!(
            hp::check_gauss_legendre_rule_hp(&nodes, &weights, p, 8)
                .unwrap()
                .checks_passed
        );
        assert!(hp::check_gauss_legendre_rule_hp(&nodes, &weights, p, 7).is_err());

        let mut bad = nodes.clone();
        for x in &mut bad {
            *x /= 2;
        }
        assert!(hp::cache_structural_check(&bad, &weights, p).is_some());
        let mut bad = weights.clone();
        bad[0] = Float::with_val(p, 0);
        bad[7] = Float::with_val(p, 0);
        assert!(hp::cache_structural_check(&nodes, &bad, p).is_some());
        let mut bad = nodes.clone();
        bad.swap(0, 1);
        assert!(hp::cache_structural_check(&bad, &weights, p).is_some());
        assert!(hp::cache_structural_check(&[], &[], p).is_some());
    }

    #[test]
    fn rule_validation_rejects_symmetric_mass_preserving_interior_corruption() {
        // Interior nodes have |P_n'| near sqrt(n), not n^2/2. The former
        // n^2-scaled tolerance accepted a central pair moved by 1e-13
        // (about 1.8e6 ulp at 64 bits) and weights traded between neighbors,
        // which keep symmetry and mass.
        let (n, p) = (500, 64);
        let check = |x: &[Float], w: &[Float], order: usize, precision: u32| {
            hp::check_gauss_legendre_rule_hp(x, w, precision, order)
                .map(|report| report.checks_passed)
                .unwrap_or(false)
        };
        let (nodes, weights) = hp::gauss_legendre_nodes(n, p, hp::CacheMode::Off);
        assert!(check(&nodes, &weights, n, p));
        let (low, high) = (n / 2 - 1, n / 2);
        let shift = Float::with_val(p, 1e-13);
        let mut bad = nodes.clone();
        bad[low] -= &shift;
        bad[high] += &shift;
        assert!(!check(&bad, &weights, n, p));
        let delta = Float::with_val(p, &weights[high] * 3e-12);
        let mut bad = weights.clone();
        bad[low] += &delta;
        bad[high] += &delta;
        bad[low - 1] -= &delta;
        bad[high + 1] -= &delta;
        assert!(!check(&nodes, &bad, n, p));
        for (order, precision) in [(1952, 64), (64, 256), (300, 1024)] {
            let (x, w) = hp::gauss_legendre_nodes(order, precision, hp::CacheMode::Off);
            assert!(check(&x, &w, order, precision), "n={order}, p={precision}");
        }
    }

    /// Real small GL rules exercise cache paths without weakening validation.
    fn valid_gl_json(n: usize) -> String {
        let (nodes, weights) = hp::gauss_legendre_nodes(n, 64, hp::CacheMode::Off);
        serde_json::json!({
            "schema_version": 1,
            "toolkit_version": hp::toolkit_version_for_test(),
            "n_pts": n,
            "precision_bits": 64,
            "nodes": nodes.iter().map(Float::to_string).collect::<Vec<_>>(),
            "weights": weights.iter().map(Float::to_string).collect::<Vec<_>>(),
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
        let json_payload = structurally_invalid_gl_json(n, prec);
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
        let zip_payload = valid_gl_json(n);
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
        let payload = valid_gl_json(n);
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
            "fresh compute should write the .json.zip to gl_cache/ under the redirected cache root"
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
    /// `gauss_legendre_compute` (no cache involved because the cache
    /// root is redirected to a fresh scratch directory).
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
        fn write_zip(path: &std::path::Path, data: &str) {
            use std::io::Write;
            let mut archive = zip::ZipWriter::new(std::fs::File::create(path).unwrap());
            let name = path
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .strip_suffix(".zip")
                .unwrap();
            archive
                .start_file(name, zip::write::SimpleFileOptions::default())
                .unwrap();
            archive.write_all(data.as_bytes()).unwrap();
            archive.finish().unwrap();
        }

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
        let valid_path = cache_dir.join("prec64_npts4.json.zip");
        write_zip(&valid_path, &valid_json);

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
        let bad_path = cache_dir.join("prec64_npts5.json.zip");
        write_zip(&bad_path, &bad_json);

        // 3. Unrecognized filename — should be reported as Skipped.
        let skipped_path = cache_dir.join("not_a_cache_file.txt");
        std::fs::write(&skipped_path, "irrelevant").unwrap();

        // 4. File matching the pattern but malformed JSON.
        let malformed_path = cache_dir.join("prec64_npts3.json.zip");
        write_zip(&malformed_path, "{");

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
                CacheFileStatus::Stale { .. } => panic!("unexpected stale fixture"),
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
        let (nodes, weights) = hp::gauss_legendre_nodes(2, 128, hp::CacheMode::Off);
        let good = serde_json::json!({
            "schema_version": 1, "toolkit_version": hp::toolkit_version_for_test(),
            "n_pts": 2, "precision_bits": 128,
            "nodes": nodes.iter().map(rug::Float::to_string).collect::<Vec<_>>(),
            "weights": weights.iter().map(rug::Float::to_string).collect::<Vec<_>>()
        });
        assert!(hp::parse_gl_json_for_test(&good.to_string(), 2, 128).is_some());
        for (field, value) in [("schema_version", 2), ("n_pts", 3), ("precision_bits", 64)] {
            let mut bad = good.clone();
            bad[field] = serde_json::json!(value);
            assert!(hp::parse_gl_json_for_test(&bad.to_string(), 2, 128).is_none());
        }
    }
}
