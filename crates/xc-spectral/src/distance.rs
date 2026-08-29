// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Weighted distances between CCM eigenfunctions and a runtime-supplied target.
//!
//! Implements the collaboration's central measurement,
//!
//! ```text
//!   d(N, λ) = ∫₁^λ |f_{N,λ}(u) − target(u)| u^{−α} du
//! ```
//!
//! together with the weighted norm `‖g‖_α = ∫₁^λ |g(u)| u^{−α} du` and the
//! inter-discretization distance `D_α(N, M; λ) = ‖f_{N,λ} − f_{M,λ}‖_α`.
//! The program objective is `lim_{λ→∞} lim_{N→∞} d(N, λ) = 0`, with the limits
//! in that order: the eigenfunction must first stabilize in `N` at fixed `λ`.
//!
//! `f_{N,λ}` is the even CCM ground-state eigenfunction reconstructed from its
//! `V_n` coefficients and normalized so `f_{N,λ}(1) = 1`. Since the target is normalized at `1`
//! exactly (see [`crate::target`]), the integrand of `d` vanishes at the left
//! endpoint.
//!
//! ## Conventions travel with results
//!
//! At finite resolution the value of every quantity here depends on the
//! integration rule, the grid variable, the resolution, and `α`. Independent
//! groups in this collaboration integrate differently, so every result type
//! records the full convention it was computed under. A number separated from
//! its convention is not comparable and should not be reported.
//!
//! `α` is an explicit parameter, never an assumption. `α = 1/2` is the
//! exponent corresponding to uniform convergence on the full critical strip,
//! while `α < 1/2` corresponds to uniform convergence only on compact
//! substrips; which one a study wants is the caller's to state.
//!
//! # Cache effects
//!
//! The measurement functions perform no cache lookup, persistence, or
//! publication. The `ccm_*` entry points resolve their eigenvector through the
//! ordinary managed CCM cache routes (reuse-first) and write nothing. The
//! `capture_*` functions are the exception and exist precisely to persist:
//! they retain their results as `ccm-distance` artifacts through the supplied
//! cache context.

use anyhow::Result;
use xc_numerics::grid_integral::{uniform_grid_integral_f64, GridVariable, UniformGridScheme};

/// How a weighted integral is evaluated.
///
/// The two families are peers. Neither is the toolkit's preferred or
/// authoritative rule, and results carry the rule that produced them so any
/// two measurements can be compared on equal footing.
///
/// Choosing between them is a property of the integrand, not a convention to
/// inherit. Gauss--Legendre converges spectrally on smooth integrands, but the
/// distance integrand carries an absolute value: at every interior sign change
/// of the signed residual its derivative has a kink, and Gauss--Legendre falls back to
/// algebraic convergence there while a composite uniform rule stays `O(h²)`.
/// Whether that matters for a given `(N, λ)` is an empirical question about
/// the sign structure of the residual, and is best answered from a retained
/// eigenfunction profile rather than assumed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WeightedIntegrationRule {
    /// Composite rule on an equally spaced grid.
    UniformGrid {
        scheme: UniformGridScheme,
        variable: GridVariable,
        /// Number of grid cells.
        steps: usize,
    },
    /// Gauss--Legendre with the given node count.
    GaussLegendre {
        points: usize,
        /// Variable the nodes are placed in. Placing them in `ln u` is a
        /// different rule from placing them in `u`, not a reparameterization
        /// of the same one.
        variable: GridVariable,
    },
}

impl WeightedIntegrationRule {
    /// Stable family identifier for recording and for semantic keys.
    pub fn family(self) -> &'static str {
        match self {
            Self::UniformGrid { .. } => "uniform_grid",
            Self::GaussLegendre { .. } => "gauss_legendre",
        }
    }

    /// Stable rule identifier, distinguishing the uniform-grid schemes from
    /// each other and from Gauss--Legendre.
    pub fn rule(self) -> &'static str {
        match self {
            Self::UniformGrid { scheme, .. } => scheme.as_str(),
            Self::GaussLegendre { .. } => "gauss_legendre",
        }
    }

    /// Variable the nodes are placed in.
    pub fn variable(self) -> GridVariable {
        match self {
            Self::UniformGrid { variable, .. } | Self::GaussLegendre { variable, .. } => variable,
        }
    }

    /// Grid cells or quadrature points, whichever the rule uses. Reported
    /// under one name because it is the rule's resolution parameter; it is
    /// not a claim that the two are numerically comparable.
    pub fn resolution(self) -> usize {
        match self {
            Self::UniformGrid { steps, .. } => steps,
            Self::GaussLegendre { points, .. } => points,
        }
    }

    fn validate(self) -> Result<()> {
        if self.resolution() == 0 {
            anyhow::bail!(
                "integration rule {} requires a positive resolution",
                self.rule()
            );
        }
        Ok(())
    }
}

/// A weighted-integral value together with the full convention that produced
/// it. Report these fields alongside the value, always.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeightedGridValueF64 {
    /// The computed integral value.
    pub value: f64,
    /// Upper integration limit `λ` (the lower limit is `1`).
    pub lambda: f64,
    /// Weight exponent: the integrand carries `u^{−α}`.
    pub alpha: f64,
    /// Rule that produced the value.
    pub rule: WeightedIntegrationRule,
}

/// Integrate `g` over `[1, λ]` under `rule`, at binary64.
fn integrate_f64<G: Fn(f64) -> f64>(
    g: G,
    lambda: f64,
    rule: WeightedIntegrationRule,
) -> Result<f64> {
    match rule {
        WeightedIntegrationRule::UniformGrid {
            scheme,
            variable,
            steps,
        } => uniform_grid_integral_f64(g, 1.0, lambda, steps, scheme, variable),
        WeightedIntegrationRule::GaussLegendre { points, variable } => {
            let (nodes, weights) = xc_numerics::quadrature::gl_nodes_weights_f64(points);
            let (lo, hi) = match variable {
                GridVariable::U => (1.0_f64, lambda),
                GridVariable::LogU => (0.0_f64, lambda.ln()),
            };
            let mid = 0.5 * (lo + hi);
            let half = 0.5 * (hi - lo);
            let mut sum = 0.0_f64;
            for (node, weight) in nodes.iter().zip(&weights) {
                let t = mid + half * node;
                sum += weight
                    * match variable {
                        GridVariable::U => g(t),
                        // du = u dt under u = e^t.
                        GridVariable::LogU => {
                            let u = t.exp();
                            g(u) * u
                        }
                    };
            }
            Ok(sum * half)
        }
    }
}

fn validate_lambda_alpha(lambda: f64, alpha: f64) -> Result<()> {
    if !lambda.is_finite() || lambda <= 1.0 {
        anyhow::bail!("weighted distances integrate over [1, λ] and need λ > 1 (got {lambda})");
    }
    if !alpha.is_finite() {
        anyhow::bail!("weight exponent α must be finite (got {alpha})");
    }
    Ok(())
}

/// `∫₁^λ |f(u)| u^{−α} du` under `rule`, at binary64.
pub fn weighted_alpha_norm_f64<F: Fn(f64) -> f64>(
    f: F,
    lambda: f64,
    alpha: f64,
    rule: WeightedIntegrationRule,
) -> Result<WeightedGridValueF64> {
    weighted_alpha_distance_f64(f, |_| 0.0, lambda, alpha, rule)
}

/// `∫₁^λ |f(u) − g(u)| u^{−α} du` under `rule`, at binary64.
///
/// With `f` and `g` two discretizations `f_{N,λ}` and `f_{M,λ}` this is
/// `D_α(N, M; λ)`; with `g` the runtime profile it is the distance to target (see
/// [`distance_to_target_f64`]).
pub fn weighted_alpha_distance_f64<F: Fn(f64) -> f64, G: Fn(f64) -> f64>(
    f: F,
    g: G,
    lambda: f64,
    alpha: f64,
    rule: WeightedIntegrationRule,
) -> Result<WeightedGridValueF64> {
    validate_lambda_alpha(lambda, alpha)?;
    rule.validate()?;
    let integrand = |u: f64| (f(u) - g(u)).abs() * u.powf(-alpha);
    let value = integrate_f64(integrand, lambda, rule)?;
    Ok(WeightedGridValueF64 {
        value,
        lambda,
        alpha,
        rule,
    })
}

/// `d(N, λ) = ∫₁^λ |f(u) − target(u)| u^{−α} du` at binary64.
pub fn distance_to_target_f64<F: Fn(f64) -> f64>(
    f: F,
    lambda: f64,
    alpha: f64,
    rule: WeightedIntegrationRule,
) -> Result<WeightedGridValueF64> {
    let target = crate::target::TargetEvaluatorF64::from_environment()?;
    weighted_alpha_distance_f64(f, |u| target.value(u), lambda, alpha, rule)
}

/// The even CCM eigenfunction reconstructed from `V_n` coefficients,
/// normalized so `f(1) = 1`, at binary64.
///
/// From the full `2N+1` coefficient vector (layout `j = −N…N`, index `N` at
/// `j = 0`; see `crate::ccm::hp::expand_even_sector_vector` for the sector
/// route producing it), the reconstruction is the even cosine sum
///
/// ```text
///   f_raw(u) = ξ₀ + 2 Σ_{n=1}^{N} ξ_n cos(2π n ln(λu) / L),   L = ln λ²
///   f(u)     = f_raw(u) / f_raw(1)
/// ```
///
/// The `1/√L` factor of the unnormalized reconstruction cancels in the
/// normalization and is omitted.
#[derive(Clone, Debug)]
pub struct WeilEigenfunctionF64 {
    xi_zero: f64,
    xi_positive: Vec<f64>,
    lambda: f64,
    log_lambda_sq: f64,
    raw_at_one: f64,
}

impl WeilEigenfunctionF64 {
    /// Build from the full `2N+1` `V_n` coefficient vector.
    ///
    /// Uses `ξ₀` and the positive-index coefficients, which is exact for the
    /// even eigenfunctions this measurement concerns.
    pub fn from_v_basis(xi: &[f64], n_modes: usize, lambda: f64) -> Result<Self> {
        if xi.len() != 2 * n_modes + 1 {
            anyhow::bail!(
                "xi has wrong length: got {}, expected 2N+1 = {}",
                xi.len(),
                2 * n_modes + 1
            );
        }
        if !lambda.is_finite() || lambda <= 1.0 {
            anyhow::bail!("eigenfunction reconstruction needs λ > 1 (got {lambda})");
        }
        let mut candidate = Self {
            xi_zero: xi[n_modes],
            xi_positive: xi[n_modes + 1..].to_vec(),
            lambda,
            log_lambda_sq: (lambda * lambda).ln(),
            raw_at_one: 1.0,
        };
        let raw_at_one = candidate.raw_eval(1.0);
        if !raw_at_one.is_finite() || raw_at_one == 0.0 {
            anyhow::bail!(
                "eigenfunction cannot be normalized: f_raw(1) = {raw_at_one}; \
                 the f(1) = 1 convention requires a nonzero value at u = 1"
            );
        }
        candidate.raw_at_one = raw_at_one;
        Ok(candidate)
    }

    /// Build from the normalized coefficients retained in a
    /// `ccm_eigenfunction_profile` artifact: `j = 0 … N`, negative indices
    /// omitted because the eigenfunction is even.
    ///
    /// This is the reader-side counterpart of the retained coefficients, and
    /// it is what makes them usable without reimplementing the reconstruction.
    /// The coefficients are mirrored into the full `2N+1` layout and passed
    /// through the same constructor as any other `V_n` vector, so the two
    /// paths cannot drift apart. Normalization is reapplied and is a no-op for
    /// already-normalized input, which also means unnormalized coefficients
    /// are accepted and normalized rather than silently mis-scaled.
    pub fn from_normalized_coefficients(coefficients: &[f64], lambda: f64) -> Result<Self> {
        if coefficients.is_empty() {
            anyhow::bail!("eigenfunction needs at least the j = 0 coefficient");
        }
        let n_modes = coefficients.len() - 1;
        let mut xi = vec![0.0_f64; 2 * n_modes + 1];
        xi[n_modes] = coefficients[0];
        for (k, value) in coefficients[1..].iter().enumerate() {
            xi[n_modes + k + 1] = *value;
            xi[n_modes - k - 1] = *value;
        }
        Self::from_v_basis(&xi, n_modes, lambda)
    }

    fn raw_eval(&self, u: f64) -> f64 {
        let phase_base = 2.0 * std::f64::consts::PI * (self.lambda * u).ln() / self.log_lambda_sq;
        let mut acc = self.xi_zero;
        for (k, xi) in self.xi_positive.iter().enumerate() {
            acc += 2.0 * xi * ((k + 1) as f64 * phase_base).cos();
        }
        acc
    }

    /// Evaluate the normalized eigenfunction; `eval(1.0) == 1.0` exactly.
    pub fn eval(&self, u: f64) -> f64 {
        self.raw_eval(u) / self.raw_at_one
    }

    /// The normalized `V_n` coefficients for `j = 0 … N`.
    ///
    /// Already divided by `f_raw(1)`, negative indices omitted because the
    /// eigenfunction is even. This is the exact input
    /// [`Self::from_normalized_coefficients`] accepts, and the binary64
    /// counterpart of `hp::WeilEigenfunction::normalized_coefficients` (that
    /// module is feature-gated, so this is deliberately not a link).
    pub fn normalized_coefficients(&self) -> Vec<f64> {
        let mut coefficients = Vec::with_capacity(self.xi_positive.len() + 1);
        coefficients.push(self.xi_zero / self.raw_at_one);
        coefficients.extend(self.xi_positive.iter().map(|value| value / self.raw_at_one));
        coefficients
    }

    /// `λ` this eigenfunction was reconstructed at.
    pub fn lambda(&self) -> f64 {
        self.lambda
    }
}

/// Where the target residual changes sign on `(1, λ]`, sampled on a grid.
///
/// Each interior sign change is a derivative kink of the distance integrand
/// its absolute value. Gauss--Legendre earns its spectral convergence only on smooth
/// integrands, so a positive crossing count is the concrete signal that a
/// Gauss--Legendre distance for this configuration converges algebraically
/// rather than spectrally, and that a composite uniform rule may be the more
/// trustworthy comparison. A zero count means the integrand is smooth and
/// Gauss--Legendre is at its best.
///
/// `u = 1` is excluded: both profiles equal `1`, so the difference vanishes
/// there by construction rather than by a crossing.
#[derive(Clone, Debug, PartialEq)]
pub struct TargetCrossingReportF64 {
    /// Sample points examined on `(1, λ]`.
    pub samples: usize,
    /// Sign of the target residual at the first sample right of `u = 1`; `0` if it
    /// vanishes there to within the detection threshold.
    pub initial_sign: i8,
    /// Brackets `(u_left, u_right)` straddling each detected sign change.
    /// A crossing between two samples is detected; one that occurs and
    /// reverses within a single cell is not, so a zero count is evidence at
    /// the sampled resolution rather than a proof of no crossing.
    pub brackets: Vec<(f64, f64)>,
}

impl TargetCrossingReportF64 {
    /// Number of detected sign changes.
    pub fn crossings(&self) -> usize {
        self.brackets.len()
    }

    /// Whether the integrand appears smooth at this sampling resolution, and
    /// therefore whether Gauss--Legendre retains its spectral advantage.
    pub fn integrand_appears_smooth(&self) -> bool {
        self.brackets.is_empty()
    }
}

/// Detect target-residual sign changes on `(1, λ]` at binary64.
pub fn target_crossings_f64<F: Fn(f64) -> f64>(
    f: F,
    lambda: f64,
    samples: usize,
    variable: GridVariable,
) -> Result<TargetCrossingReportF64> {
    if !lambda.is_finite() || lambda <= 1.0 {
        anyhow::bail!("crossing detection needs λ > 1 (got {lambda})");
    }
    if samples < 2 {
        anyhow::bail!("crossing detection needs at least two samples");
    }
    let target = crate::target::TargetEvaluatorF64::from_environment()?;
    let difference = |u: f64| f(u) - target.value(u);
    let (lo, hi) = match variable {
        GridVariable::U => (1.0_f64, lambda),
        GridVariable::LogU => (0.0_f64, lambda.ln()),
    };
    let step = (hi - lo) / samples as f64;
    let point = |index: usize| {
        let t = lo + step * index as f64;
        match variable {
            GridVariable::U => t,
            GridVariable::LogU => t.exp(),
        }
    };
    let mut brackets = Vec::new();
    let mut initial_sign = 0_i8;
    let mut previous: Option<(f64, f64)> = None;
    // Start at index 1: index 0 is u = 1, where the difference vanishes by
    // construction and carries no sign information.
    for index in 1..=samples {
        let u = point(index);
        let value = difference(u);
        let sign = if value > 0.0 {
            1_i8
        } else if value < 0.0 {
            -1
        } else {
            0
        };
        if initial_sign == 0 {
            initial_sign = sign;
        }
        if let Some((previous_u, previous_value)) = previous {
            let previous_sign = if previous_value > 0.0 {
                1_i8
            } else if previous_value < 0.0 {
                -1
            } else {
                0
            };
            if previous_sign != 0 && sign != 0 && previous_sign != sign {
                brackets.push((previous_u, u));
            }
        }
        previous = Some((u, value));
    }
    Ok(TargetCrossingReportF64 {
        samples,
        initial_sign,
        brackets,
    })
}

#[cfg(feature = "hp")]
pub mod hp {
    //! High-precision weighted distances via rug/MPFR.

    use super::WeightedIntegrationRule;
    use anyhow::Result;
    use rug::{float::Constant, Float, Integer};
    use serde::{Deserialize, Serialize};
    use xc_numerics::grid_integral::{hp::uniform_grid_integral, GridVariable, UniformGridScheme};

    /// Guard bits for internal evaluation above the requested precision.
    const GUARD_BITS: u32 = 64;
    const RESIDUAL_MASS_CONSISTENCY_POLICY: &str =
        "snap_signed_to_absolute_within_scaled_2^(-(precision_bits-8));otherwise_reject_v1";

    /// A weighted-integral value with the full convention that produced it.
    #[derive(Clone, Debug)]
    pub struct WeightedGridValueHp {
        /// The computed integral value at the requested precision.
        pub value: Float,
        /// Upper integration limit `λ` (the lower limit is `1`).
        pub lambda: Float,
        /// Weight exponent: the integrand carries `u^{−α}`.
        pub alpha: Float,
        /// Rule that produced the value.
        pub rule: WeightedIntegrationRule,
        /// Requested working precision in bits.
        pub precision_bits: u32,
    }

    fn validate_lambda_alpha(lambda: &Float, alpha: &Float) -> Result<()> {
        if !lambda.is_finite() || *lambda <= 1u32 {
            anyhow::bail!(
                "weighted distances integrate over [1, λ] and need λ > 1 (got {})",
                lambda.to_f64()
            );
        }
        if !alpha.is_finite() {
            anyhow::bail!("weight exponent α must be finite");
        }
        Ok(())
    }

    /// `u^{−α}` as `exp(−α ln u)`, valid for `u ≥ 1`.
    fn weight(u: &Float, alpha: &Float, working: u32) -> Float {
        let mut exponent = Float::with_val(working, u.clone().ln());
        exponent *= alpha;
        exponent.neg_assign();
        exponent.exp()
    }
    use rug::ops::NegAssign;

    /// Gauss--Legendre tables shared across the measurements of one capture
    /// call, keyed by `(points, working_precision)`.
    ///
    /// Ordinary measurement APIs construct these tables cache-off. Managed
    /// capture APIs may preload the same tables through the typed quadrature
    /// artifact fabric, allowing exact reuse across configurations while the
    /// integration arithmetic and reduction order remain unchanged.
    pub(crate) struct SharedGlTables(
        std::collections::HashMap<(usize, u32), (Vec<Float>, Vec<Float>)>,
    );

    impl SharedGlTables {
        pub(crate) fn new() -> Self {
            Self(std::collections::HashMap::new())
        }

        pub(crate) fn preload_managed(
            &mut self,
            rules: &[WeightedIntegrationRule],
            precision_bits: u32,
            cache: &xc_cache::ArtifactCacheContext<'_>,
        ) -> Result<()> {
            // Disabled and deliberately read-only non-reuse contexts preserve
            // the standalone cache-off behavior. RequireReuse is allowed
            // through because a missing quadrature artifact must fail rather
            // than silently reconstruct the table.
            if !cache.mode.fabric_enabled()
                || (!cache.write_on_miss && !cache.mode.requires_reuse())
            {
                return Ok(());
            }
            let working = precision_bits.saturating_add(GUARD_BITS);
            for rule in rules {
                let WeightedIntegrationRule::GaussLegendre { points, .. } = *rule else {
                    continue;
                };
                if self.0.contains_key(&(points, working)) {
                    continue;
                }
                let request = xc_cache::ArtifactCacheContext {
                    resolver: cache.resolver,
                    reference_resolver: cache.reference_resolver,
                    acceptance: cache.acceptance,
                    ordered_overlays: cache.ordered_overlays.clone(),
                    mode: cache.mode,
                    write_on_miss: cache.write_on_miss,
                    write_visibility: cache.write_visibility,
                    requested_assurance: cache.requested_assurance,
                    certification_failure_policy: cache.certification_failure_policy,
                    production_sink: cache.production_sink,
                };
                let resolved = xc_numerics::quadrature::gauss_legendre_nodes_via_cache(
                    points, working, request,
                )?;
                self.0
                    .insert((points, working), (resolved.nodes, resolved.weights));
            }
            Ok(())
        }

        fn get_or_build(&mut self, points: usize, working: u32) -> &(Vec<Float>, Vec<Float>) {
            self.0.entry((points, working)).or_insert_with(|| {
                xc_numerics::quadrature::gauss_legendre_nodes(
                    points,
                    working,
                    xc_numerics::quadrature::CacheMode::Off,
                )
            })
        }
    }

    /// Integrate `g` over `[1, λ]` under `rule` at `prec` bits.
    ///
    /// Ordinary callers compute Gauss--Legendre nodes with the quadrature
    /// cache disabled. When `tables` is supplied, a table already built or
    /// resolved for the same `(points, working)` request is reused instead of
    /// reconstructed.
    fn integrate_with<G: Fn(&Float) -> Float>(
        g: G,
        lambda: &Float,
        rule: WeightedIntegrationRule,
        prec: u32,
        tables: Option<&mut SharedGlTables>,
    ) -> Result<Float> {
        let working = prec.saturating_add(GUARD_BITS);
        match rule {
            WeightedIntegrationRule::UniformGrid {
                scheme,
                variable,
                steps,
            } => {
                let one = Float::with_val(working, 1u32);
                uniform_grid_integral(g, &one, lambda, steps, scheme, variable, prec)
            }
            WeightedIntegrationRule::GaussLegendre { points, variable } => {
                let built;
                let (nodes, weights): (&Vec<Float>, &Vec<Float>) = match tables {
                    Some(tables) => {
                        let table = tables.get_or_build(points, working);
                        (&table.0, &table.1)
                    }
                    None => {
                        built = xc_numerics::quadrature::gauss_legendre_nodes(
                            points,
                            working,
                            xc_numerics::quadrature::CacheMode::Off,
                        );
                        (&built.0, &built.1)
                    }
                };
                let (lo, hi) = match variable {
                    GridVariable::U => (
                        Float::with_val(working, 1u32),
                        Float::with_val(working, lambda),
                    ),
                    GridVariable::LogU => (
                        Float::with_val(working, 0u32),
                        Float::with_val(working, lambda).ln(),
                    ),
                };
                let mut mid = Float::with_val(working, &lo + &hi);
                mid /= 2u32;
                let mut half = Float::with_val(working, &hi - &lo);
                half /= 2u32;
                let mut sum = Float::with_val(working, 0u32);
                for (node, weight_value) in nodes.iter().zip(weights.iter()) {
                    let mut point = half.clone();
                    point *= node;
                    point += &mid;
                    let mut term = match variable {
                        GridVariable::U => g(&point),
                        // du = u dt under u = e^t.
                        GridVariable::LogU => {
                            let u = point.exp();
                            g(&u) * u
                        }
                    };
                    term *= weight_value;
                    sum += term;
                }
                sum *= &half;
                Ok(Float::with_val(prec, sum))
            }
        }
    }

    /// `∫₁^λ |f(u)| u^{−α} du` under `rule` at `prec` bits.
    pub fn weighted_alpha_norm<F: Fn(&Float) -> Float>(
        f: F,
        lambda: &Float,
        alpha: &Float,
        rule: WeightedIntegrationRule,
        prec: u32,
    ) -> Result<WeightedGridValueHp> {
        weighted_alpha_norm_with_tables(f, lambda, alpha, rule, prec, None)
    }

    /// [`weighted_alpha_norm`] with optional shared Gauss--Legendre tables.
    pub(crate) fn weighted_alpha_norm_with_tables<F: Fn(&Float) -> Float>(
        f: F,
        lambda: &Float,
        alpha: &Float,
        rule: WeightedIntegrationRule,
        prec: u32,
        tables: Option<&mut SharedGlTables>,
    ) -> Result<WeightedGridValueHp> {
        let working = prec.saturating_add(GUARD_BITS);
        let zero = move |_: &Float| Float::with_val(working, 0u32);
        weighted_alpha_distance_with_tables(f, zero, lambda, alpha, rule, prec, tables)
    }

    /// `∫₁^λ |f(u) − g(u)| u^{−α} du` under `rule` at `prec` bits.
    ///
    /// With `f` and `g` two discretizations `f_{N,λ}` and `f_{M,λ}` this is
    /// `D_α(N, M; λ)`. The self-distance of any function is exactly zero.
    pub fn weighted_alpha_distance<F: Fn(&Float) -> Float, G: Fn(&Float) -> Float>(
        f: F,
        g: G,
        lambda: &Float,
        alpha: &Float,
        rule: WeightedIntegrationRule,
        prec: u32,
    ) -> Result<WeightedGridValueHp> {
        weighted_alpha_distance_with_tables(f, g, lambda, alpha, rule, prec, None)
    }

    /// [`weighted_alpha_distance`] with optional shared Gauss--Legendre
    /// tables. The arithmetic is identical to the per-call path; only table
    /// construction frequency changes.
    pub(crate) fn weighted_alpha_distance_with_tables<
        F: Fn(&Float) -> Float,
        G: Fn(&Float) -> Float,
    >(
        f: F,
        g: G,
        lambda: &Float,
        alpha: &Float,
        rule: WeightedIntegrationRule,
        prec: u32,
        tables: Option<&mut SharedGlTables>,
    ) -> Result<WeightedGridValueHp> {
        validate_lambda_alpha(lambda, alpha)?;
        rule.validate()?;
        let working = prec.saturating_add(GUARD_BITS);
        let alpha_working = Float::with_val(working, alpha);
        let integrand = |u: &Float| {
            let mut difference = f(u);
            difference -= g(u);
            difference.abs_mut();
            difference * weight(u, &alpha_working, working)
        };
        let value = integrate_with(integrand, lambda, rule, prec, tables)?;
        Ok(WeightedGridValueHp {
            value,
            lambda: Float::with_val(prec, lambda),
            alpha: Float::with_val(prec, alpha),
            rule,
            precision_bits: prec,
        })
    }

    /// `d(N, λ) = ∫₁^λ |f(u) − target(u)| u^{−α} du` at `prec` bits.
    pub fn distance_to_target<F: Fn(&Float) -> Float>(
        f: F,
        lambda: &Float,
        alpha: &Float,
        rule: WeightedIntegrationRule,
        prec: u32,
    ) -> Result<WeightedGridValueHp> {
        distance_to_target_with_tables(f, lambda, alpha, rule, prec, None)
    }

    /// [`distance_to_target`] with optional shared Gauss--Legendre tables.
    pub(crate) fn distance_to_target_with_tables<F: Fn(&Float) -> Float>(
        f: F,
        lambda: &Float,
        alpha: &Float,
        rule: WeightedIntegrationRule,
        prec: u32,
        tables: Option<&mut SharedGlTables>,
    ) -> Result<WeightedGridValueHp> {
        distance_to_target_with_tables_bound(f, lambda, alpha, rule, prec, tables, None)
    }

    #[allow(clippy::too_many_arguments)]
    fn distance_to_target_with_tables_bound<F: Fn(&Float) -> Float>(
        f: F,
        lambda: &Float,
        alpha: &Float,
        rule: WeightedIntegrationRule,
        prec: u32,
        tables: Option<&mut SharedGlTables>,
        expected_target_digest: Option<&str>,
    ) -> Result<WeightedGridValueHp> {
        let working = prec.saturating_add(GUARD_BITS);
        let target = crate::target::hp::TargetEvaluator::from_environment(working)?;
        if expected_target_digest.is_some_and(|expected| expected != target.definition_digest()) {
            anyhow::bail!(
                "runtime target specification changed after its semantic identity was fixed"
            );
        }
        weighted_alpha_distance_with_tables(
            f,
            |u| target.value(u),
            lambda,
            alpha,
            rule,
            prec,
            tables,
        )
    }

    /// Signed counterpart of [`distance_to_target_with_tables`]. This keeps
    /// the same nodes, weights, target, and reduction order but does not apply
    /// the absolute value to the target residual.
    fn signed_residual_to_target_with_tables<F: Fn(&Float) -> Float>(
        f: F,
        lambda: &Float,
        alpha: &Float,
        rule: WeightedIntegrationRule,
        prec: u32,
        tables: Option<&mut SharedGlTables>,
        expected_target_digest: Option<&str>,
    ) -> Result<Float> {
        validate_lambda_alpha(lambda, alpha)?;
        rule.validate()?;
        let working = prec.saturating_add(GUARD_BITS);
        let target = crate::target::hp::TargetEvaluator::from_environment(working)?;
        if expected_target_digest.is_some_and(|expected| expected != target.definition_digest()) {
            anyhow::bail!(
                "runtime target specification changed after its semantic identity was fixed"
            );
        }
        let alpha_working = Float::with_val(working, alpha);
        let integrand = |u: &Float| {
            let mut residual = f(u);
            residual -= target.value(u);
            residual * weight(u, &alpha_working, working)
        };
        integrate_with(integrand, lambda, rule, prec, tables)
    }

    /// The even CCM eigenfunction at HP, normalized so `f(1) = 1`.
    ///
    /// HP counterpart of [`super::WeilEigenfunctionF64`]; the reconstruction
    /// and normalization conventions are identical.
    #[derive(Clone, Debug)]
    pub struct WeilEigenfunction {
        xi_zero: Float,
        xi_positive: Vec<Float>,
        lambda: Float,
        log_lambda_sq: Float,
        raw_at_one: Float,
        working: u32,
    }

    impl WeilEigenfunction {
        /// Build from the full `2N+1` `V_n` coefficient vector at `prec` bits.
        pub fn from_v_basis(
            xi: &[Float],
            n_modes: usize,
            lambda: &Float,
            prec: u32,
        ) -> Result<Self> {
            if xi.len() != 2 * n_modes + 1 {
                anyhow::bail!(
                    "xi has wrong length: got {}, expected 2N+1 = {}",
                    xi.len(),
                    2 * n_modes + 1
                );
            }
            if !lambda.is_finite() || *lambda <= 1u32 {
                anyhow::bail!(
                    "eigenfunction reconstruction needs λ > 1 (got {})",
                    lambda.to_f64()
                );
            }
            let working = prec.saturating_add(GUARD_BITS);
            let lambda = Float::with_val(working, lambda);
            let log_lambda_sq = lambda.clone().square().ln();
            let mut candidate = Self {
                xi_zero: Float::with_val(working, &xi[n_modes]),
                xi_positive: xi[n_modes + 1..]
                    .iter()
                    .map(|value| Float::with_val(working, value))
                    .collect(),
                lambda,
                log_lambda_sq,
                raw_at_one: Float::with_val(working, 1u32),
                working,
            };
            let raw_at_one = candidate.raw_eval(&Float::with_val(working, 1u32));
            if !raw_at_one.is_finite() || raw_at_one == 0u32 {
                anyhow::bail!(
                    "eigenfunction cannot be normalized: f_raw(1) = {}; \
                     the f(1) = 1 convention requires a nonzero value at u = 1",
                    raw_at_one.to_f64()
                );
            }
            candidate.raw_at_one = raw_at_one;
            Ok(candidate)
        }

        /// Build from the normalized coefficients retained in a
        /// `ccm_eigenfunction_profile` artifact: `j = 0 … N`, negative
        /// indices omitted because the eigenfunction is even.
        ///
        /// HP counterpart of
        /// [`super::WeilEigenfunctionF64::from_normalized_coefficients`]. The
        /// coefficients are mirrored into the full `2N+1` layout and passed
        /// through the same constructor as any other `V_n` vector, so the
        /// retained artifact and a freshly solved eigenstate cannot drift
        /// apart.
        pub fn from_normalized_coefficients(
            coefficients: &[Float],
            lambda: &Float,
            prec: u32,
        ) -> Result<Self> {
            if coefficients.is_empty() {
                anyhow::bail!("eigenfunction needs at least the j = 0 coefficient");
            }
            let n_modes = coefficients.len() - 1;
            let working = prec.saturating_add(GUARD_BITS);
            let mut xi: Vec<Float> = (0..(2 * n_modes + 1))
                .map(|_| Float::with_val(working, 0u32))
                .collect();
            xi[n_modes] = Float::with_val(working, &coefficients[0]);
            for (k, value) in coefficients[1..].iter().enumerate() {
                xi[n_modes + k + 1] = Float::with_val(working, value);
                xi[n_modes - k - 1] = Float::with_val(working, value);
            }
            Self::from_v_basis(&xi, n_modes, lambda, prec)
        }

        /// The unnormalized even cosine sum at `u`.
        ///
        /// This is the hot path of every distance measurement: one
        /// high-precision `cos` per retained coefficient, at every quadrature
        /// node and every profile abscissa. The terms are independent, so they
        /// are evaluated in parallel and then folded in coefficient order.
        ///
        /// The fold is deliberately sequential. A parallel reduction would
        /// re-associate the additions and move the low bits, so retained
        /// distances would stop replaying against artifacts computed before
        /// this change; materializing the terms first and folding them in
        /// order keeps the arithmetic bit-identical to the serial form.
        ///
        /// Parallelism is applied here, on a concrete type, rather than around
        /// the quadrature loops: those run through a generic integrand, and
        /// requiring `Sync` on it triggers an internal compiler error in
        /// rustc 1.95.0 (see the note on `integrate_with`).
        fn raw_eval(&self, u: &Float) -> Float {
            use rayon::prelude::*;

            let two_pi = Float::with_val(self.working, Constant::Pi) * 2u32;
            let mut phase_base = Float::with_val(self.working, &self.lambda * u).ln();
            phase_base *= two_pi;
            phase_base /= &self.log_lambda_sq;
            let terms: Vec<Float> = self
                .xi_positive
                .par_iter()
                .enumerate()
                .map(|(k, xi)| {
                    let mut angle = phase_base.clone();
                    angle *= (k + 1) as u32;
                    let mut term = angle.cos();
                    term *= xi;
                    term *= 2u32;
                    term
                })
                .collect();
            let mut acc = self.xi_zero.clone();
            for term in terms {
                acc += term;
            }
            acc
        }

        /// Evaluate the normalized eigenfunction; `eval(1) == 1` exactly.
        pub fn eval(&self, u: &Float) -> Float {
            let mut value = self.raw_eval(u);
            value /= &self.raw_at_one;
            value
        }

        /// The normalized `V_n` coefficients for `j = 0 … N`.
        ///
        /// Already divided by `f_raw(1)`, so evaluating the even cosine sum
        /// with these directly yields `f`. The negative indices are omitted
        /// because the eigenfunction is even. This is the lossless form of the
        /// eigenfunction: from it any rule, resolution, or abscissa can be
        /// recomputed exactly, which a sampled profile cannot support.
        pub fn normalized_coefficients(&self) -> Vec<Float> {
            let mut coefficients = Vec::with_capacity(self.xi_positive.len() + 1);
            let mut zero = self.xi_zero.clone();
            zero /= &self.raw_at_one;
            coefficients.push(zero);
            for value in &self.xi_positive {
                let mut scaled = value.clone();
                scaled /= &self.raw_at_one;
                coefficients.push(scaled);
            }
            coefficients
        }

        /// `λ` this eigenfunction was reconstructed at.
        pub fn lambda(&self) -> &Float {
            &self.lambda
        }
    }

    /// Where the target residual changes sign on `(1, λ]`, sampled at high precision.
    ///
    /// HP counterpart of [`super::TargetCrossingReportF64`]. Crossing
    /// detection belongs at working precision as much as the distance does:
    /// the question it answers — whether the absolute residual has an interior derivative
    /// kink, and therefore whether a Gauss--Legendre distance converges
    /// spectrally or only algebraically — is about the same eigenfunction the
    /// campaign measures at 500 to 7000 bits.
    #[derive(Clone, Debug)]
    pub struct TargetCrossingReport {
        /// Sample points examined on `(1, λ]`.
        pub samples: usize,
        /// Sign of the target residual at the first sample right of `u = 1`; `0` if it
        /// vanishes there.
        pub initial_sign: i8,
        /// Brackets `(u_left, u_right)` straddling each detected sign change.
        pub brackets: Vec<(Float, Float)>,
    }

    impl TargetCrossingReport {
        /// Number of detected sign changes.
        pub fn crossings(&self) -> usize {
            self.brackets.len()
        }

        /// Whether the integrand appears smooth at this sampling resolution,
        /// and therefore whether Gauss--Legendre retains its spectral
        /// advantage.
        pub fn integrand_appears_smooth(&self) -> bool {
            self.brackets.is_empty()
        }
    }

    /// Detect target-residual sign changes on `(1, λ]` at `prec` bits.
    ///
    /// `u = 1` is excluded: both normalized profiles equal `1`, so the difference
    /// vanishes there by construction rather than by a crossing. A zero count
    /// is evidence at the sampled resolution, not a proof that no crossing
    /// exists.
    pub fn target_crossings<F: Fn(&Float) -> Float>(
        f: F,
        lambda: &Float,
        samples: usize,
        variable: GridVariable,
        prec: u32,
    ) -> Result<TargetCrossingReport> {
        if !lambda.is_finite() || *lambda <= 1u32 {
            anyhow::bail!("crossing detection needs λ > 1");
        }
        if samples < 2 {
            anyhow::bail!("crossing detection needs at least two samples");
        }
        let working = prec.saturating_add(GUARD_BITS);
        let target = crate::target::hp::TargetEvaluator::from_environment(working)?;
        let (lo, hi) = match variable {
            GridVariable::U => (
                Float::with_val(working, 1u32),
                Float::with_val(working, lambda),
            ),
            GridVariable::LogU => (
                Float::with_val(working, 0u32),
                Float::with_val(working, lambda).ln(),
            ),
        };
        let mut step = Float::with_val(working, &hi - &lo);
        step /= samples as u32;

        let mut brackets = Vec::new();
        let mut initial_sign = 0_i8;
        let mut previous: Option<(Float, i8)> = None;
        for index in 1..=samples {
            let mut point = step.clone();
            point *= index as u32;
            point += &lo;
            let u = match variable {
                GridVariable::U => point,
                GridVariable::LogU => point.exp(),
            };
            let mut difference = f(&u);
            difference -= target.value(&u);
            let sign = match difference.cmp0() {
                Some(std::cmp::Ordering::Greater) => 1_i8,
                Some(std::cmp::Ordering::Less) => -1,
                _ => 0,
            };
            if initial_sign == 0 {
                initial_sign = sign;
            }
            if let Some((previous_u, previous_sign)) = &previous {
                if *previous_sign != 0 && sign != 0 && *previous_sign != sign {
                    brackets.push((previous_u.clone(), u.clone()));
                }
            }
            previous = Some((u, sign));
        }
        Ok(TargetCrossingReport {
            samples,
            initial_sign,
            brackets,
        })
    }

    /// End-to-end distance to target for one CCM configuration.
    #[derive(Clone, Debug)]
    pub struct CcmTargetDistanceHp {
        /// `λ²` the measurement was taken at.
        pub lambda_squared: f64,
        /// Mode cutoff `N`.
        pub n_modes: usize,
        /// Smallest even-sector Weil eigenvalue at this `(λ², N)`.
        pub eigenvalue: Float,
        /// One `d(N, λ)` per requested rule, each carrying its own
        /// convention. Report them together: the spread between rules is the
        /// convention sensitivity of the measurement.
        pub distances: Vec<WeightedGridValueHp>,
    }

    /// Measure `d(N, λ)` end to end for one CCM configuration.
    ///
    /// Computes (or resolves from cache, reuse-first) the even-parity Weil
    /// ground state for `params`, expands it out of the sector basis via
    /// [`crate::ccm::hp::expand_even_sector_vector`], normalizes to
    /// `f(1) = 1`, and integrates against the runtime target under the stated
    /// quadrature convention.
    ///
    /// This is a finite-`N`, finite-precision measurement. It does not on its
    /// own establish anything about either limit.
    pub fn ccm_distance_to_target_hp(
        params: &crate::ccm::CcmParams,
        cfg: &crate::ccm::hp::HighPrecConfig,
        alpha: &Float,
        rule: WeightedIntegrationRule,
    ) -> Result<CcmTargetDistanceHp> {
        let prec = cfg.precision_bits;
        let resolved = resolve_even_ground_eigenfunction(params, cfg)?;
        let distance = distance_to_target(
            |u: &Float| resolved.eigenfunction.eval(u),
            &resolved.lambda,
            alpha,
            rule,
            prec,
        )?;
        Ok(CcmTargetDistanceHp {
            lambda_squared: params.lambda_squared(),
            n_modes: params.n_modes,
            eigenvalue: resolved.eigenvalue,
            distances: vec![distance],
        })
    }

    /// The even Weil ground state for one configuration, reconstructed as a
    /// normalized eigenfunction.
    pub(crate) struct ResolvedGroundEigenfunction {
        pub eigenfunction: WeilEigenfunction,
        pub lambda: Float,
        pub eigenvalue: Float,
    }

    fn ground_eigenfunction_from_canonical_state(
        params: &crate::ccm::CcmParams,
        cfg: &crate::ccm::hp::HighPrecConfig,
        eigenvalue: &Float,
        eigenvector: &[Float],
    ) -> Result<ResolvedGroundEigenfunction> {
        let prec = cfg.precision_bits;
        let working = prec.saturating_add(GUARD_BITS);
        let lambda_sq = if params.lambda_sq.is_integer {
            Float::with_val(working, params.lambda_sq.value_u64)
        } else {
            Float::with_val(working, params.lambda_sq.value_f64)
        };
        if lambda_sq <= 1u32 {
            anyhow::bail!(
                "weighted distances integrate over [1, lambda] and need lambda^2 > 1 (got {})",
                params.lambda_squared()
            );
        }
        let expected = params.matrix_size();
        if eigenvector.len() != expected {
            anyhow::bail!(
                "canonical CCM even eigenvector has dimension {}, expected 2N+1 = {expected}",
                eigenvector.len(),
            );
        }
        let lambda = lambda_sq.sqrt();
        let eigenfunction =
            WeilEigenfunction::from_v_basis(eigenvector, params.n_modes, &lambda, prec)?;
        Ok(ResolvedGroundEigenfunction {
            eigenvalue: Float::with_val(prec, eigenvalue),
            eigenfunction,
            lambda,
        })
    }

    /// Resolve (or compute, reuse-first) the even Weil ground state and
    /// reconstruct its normalized eigenfunction.
    ///
    /// Callers evaluating several rules or several quantities against one
    /// configuration should resolve once and reuse: manifest validation and
    /// eigenvector decoding dominate at large `N` and high precision even when
    /// every resolution is a cache hit.
    pub(crate) fn resolve_even_ground_eigenfunction(
        params: &crate::ccm::CcmParams,
        cfg: &crate::ccm::hp::HighPrecConfig,
    ) -> Result<ResolvedGroundEigenfunction> {
        let working = cfg.precision_bits.saturating_add(GUARD_BITS);
        // Follow the documented LambdaSq promotion rule so an integer λ² stays
        // exact instead of round-tripping through f64.
        let lambda_sq = if params.lambda_sq.is_integer {
            Float::with_val(working, params.lambda_sq.value_u64)
        } else {
            Float::with_val(working, params.lambda_sq.value_f64)
        };
        if lambda_sq <= 1u32 {
            anyhow::bail!(
                "weighted distances integrate over [1, λ] and need λ² > 1 (got {})",
                params.lambda_squared()
            );
        }
        let mut even_cfg = cfg.clone();
        even_cfg.set_parity_policy(crate::ccm::hp::CcmParityPolicy::EvenSector);
        let state = crate::ccm::hp::build_source(params, &even_cfg)?;
        ground_eigenfunction_from_canonical_state(
            params,
            &even_cfg,
            &state.weil_min_eigenvalue,
            &state.xi,
        )
    }

    /// `D_α(N, M; λ)` for two discretizations of the same `λ²`.
    #[derive(Clone, Debug)]
    pub struct CcmDiscretizationDistanceHp {
        /// `λ²` shared by both discretizations.
        pub lambda_squared: f64,
        /// Mode cutoff of the first configuration.
        pub n_modes: usize,
        /// Mode cutoff of the second configuration.
        pub m_modes: usize,
        /// One `D_α(N, M; λ)` per requested rule.
        pub distances: Vec<WeightedGridValueHp>,
    }

    /// Measure `D_α(N, M; λ) = ‖f_{N,λ} − f_{M,λ}‖_α` end to end.
    ///
    /// This is the quantity the first stage of the program is stated in: the
    /// eigenfunction is said to stabilize at fixed `λ` when successive
    /// `D_α(N, M; λ)` shrink. It requires no target function, so it can be
    /// measured before any runtime target enters the comparison.
    ///
    /// Both configurations must share `λ²`; comparing across different `λ`
    /// would compare functions defined on different domains. The self-distance
    /// of a configuration with itself is exactly zero.
    pub fn ccm_discretization_distance_hp(
        first: &crate::ccm::CcmParams,
        second: &crate::ccm::CcmParams,
        cfg: &crate::ccm::hp::HighPrecConfig,
        alpha: &Float,
        rules: &[WeightedIntegrationRule],
    ) -> Result<CcmDiscretizationDistanceHp> {
        if rules.is_empty() {
            anyhow::bail!("discretization distance requires at least one integration rule");
        }
        // Fail before any eigenstate resolution or quadrature construction.
        for rule in rules {
            rule.validate()?;
        }
        let same_lambda = first.lambda_sq.is_integer == second.lambda_sq.is_integer
            && if first.lambda_sq.is_integer {
                first.lambda_sq.value_u64 == second.lambda_sq.value_u64
            } else {
                first.lambda_sq.value_f64 == second.lambda_sq.value_f64
            };
        if !same_lambda {
            anyhow::bail!(
                "D_alpha compares two discretizations of one lambda^2; got {} and {}",
                first.lambda_squared(),
                second.lambda_squared()
            );
        }
        let prec = cfg.precision_bits;
        let lower = resolve_even_ground_eigenfunction(first, cfg)?;
        let upper = resolve_even_ground_eigenfunction(second, cfg)?;
        let mut gl_tables = SharedGlTables::new();
        let mut distances = Vec::with_capacity(rules.len());
        for rule in rules {
            distances.push(weighted_alpha_distance_with_tables(
                |u: &Float| lower.eigenfunction.eval(u),
                |u: &Float| upper.eigenfunction.eval(u),
                &lower.lambda,
                alpha,
                *rule,
                prec,
                Some(&mut gl_tables),
            )?);
        }
        Ok(CcmDiscretizationDistanceHp {
            lambda_squared: first.lambda_squared(),
            n_modes: first.n_modes,
            m_modes: second.n_modes,
            distances,
        })
    }

    // =======================================================================
    // Persisted `ccm-distance` artifacts.
    //
    // Distance capture is opt-in. When requested, the eigenfunction profile
    // and the target-distance measurement are written as ordinary cache
    // artifacts so that downstream analysis — by other authors, or by
    // automated study of a published artifact repository — can reuse them
    // without repeating the spectral solve.
    //
    // The quadrature convention is part of the semantic key. Two runs that
    // differ only in rule, grid variable, resolution, or alpha are
    // different artifacts, never the same artifact recomputed.
    // =======================================================================

    /// Portable eigenfunction profile: `f_{N,λ}` sampled on a stated grid.
    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct PortableEigenfunctionProfile {
        pub schema_version: u32,
        pub lambda_squared: String,
        pub n_modes: usize,
        pub precision_bits: u32,
        pub grid_variable: String,
        pub sample_count: usize,
        pub normalization: String,
        /// Sample abscissae, decimal, ascending from `u = 1`.
        pub u_values: Vec<String>,
        /// `f_{N,λ}(u)` at each abscissa, same order.
        pub f_values: Vec<String>,
        /// Normalized `V_n` coefficients for `j = 0 … N`, already divided by
        /// `f_raw(1)`; negative indices are omitted because the eigenfunction
        /// is even. These are lossless: a consumer can evaluate `f` at any
        /// abscissa and therefore apply any integration rule at any
        /// resolution, rather than being limited to the rules captured here.
        pub normalized_coefficients: Vec<String>,
    }

    /// One measurement of the same configuration under one integration rule.
    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct PortableRuleMeasurement {
        /// Rule family: `uniform_grid` or `gauss_legendre`.
        pub rule_family: String,
        /// Specific rule within the family.
        pub quadrature_rule: String,
        pub grid_variable: String,
        /// Grid cells or quadrature points, per the rule.
        pub resolution: usize,
        /// `d(N, λ) = ∫₁^λ |f − target| u^{−α} du` under this rule.
        pub distance_to_target: String,
        /// `‖f_{N,λ}‖_α` under this rule.
        pub eigenfunction_norm: String,
    }

    /// Portable target-distance measurements for one configuration.
    ///
    /// Several rules are recorded together deliberately. A single distance
    /// carries no indication of how much of its value is quadrature
    /// convention, and a reader who sees one number has no prompt to ask.
    /// The spread across rules bounds the convention sensitivity directly; it
    /// is not a convergence estimate, since the entries are different rules
    /// rather than one rule at increasing resolution.
    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct PortableTargetDistance {
        pub schema_version: u32,
        /// SHA-256 of the canonical private runtime target specification.
        pub target_definition_digest: String,
        pub lambda_squared: String,
        pub n_modes: usize,
        pub precision_bits: u32,
        pub alpha: String,
        /// One entry per requested rule, in the order requested.
        pub measurements: Vec<PortableRuleMeasurement>,
        /// Smallest even-sector Weil eigenvalue at this `(λ², N)`.
        pub eigenvalue: String,
    }

    const RESOLUTION_EVIDENCE_THRESHOLD_DECADES: [u32; 3] = [15, 30, 45];
    const RESOLUTION_EVIDENCE_REFINEMENT_FACTOR: usize = 2;
    const RESOLUTION_EVIDENCE_MAXIMUM_MULTIPLIER: usize = 4;
    const RESOLUTION_EVIDENCE_RELATIVE_TOLERANCE: &str = "1e-8";

    /// Coefficient-tail diagnostics at one declared absolute threshold.
    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct PortableCoefficientTailEvidence {
        /// Absolute coefficient threshold used to define effective bandwidth.
        pub threshold: String,
        /// Largest retained nonnegative mode whose coefficient magnitude is
        /// strictly greater than `threshold`; `None` means no coefficient
        /// crossed it.
        pub effective_bandwidth: Option<usize>,
        /// One-sided `j = 0 ... N` L1 norm beyond the effective bandwidth.
        pub discarded_one_sided_l1: String,
        /// Conservative pointwise contribution bound after restoring the
        /// factor two on every positive-index cosine coefficient.
        pub discarded_cosine_pointwise_bound: String,
        /// Weighted coefficient L2 norm of the discarded even cosine series:
        /// `sqrt(c_0^2 + 2 sum_{n>0} c_n^2)` over discarded indices.
        pub discarded_cosine_l2: String,
    }

    /// Same-rule refinement evidence for one uniform-grid distance.
    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct PortableRuleResolutionEvidence {
        pub rule_family: String,
        pub quadrature_rule: String,
        pub grid_variable: String,
        pub base_resolution: usize,
        pub base_distance: String,
        pub twice_resolution: usize,
        pub twice_distance: String,
        pub q_to_2q_absolute_difference: String,
        pub q_to_2q_relative_difference: String,
        /// Present only when the Q/2Q difference exceeded the policy
        /// tolerance and the deterministic 4Q continuation was required.
        pub four_times_resolution: Option<usize>,
        pub four_times_distance: Option<String>,
        /// Difference for the last adjacent pair attempted: Q/2Q when that
        /// passed, otherwise 2Q/4Q.
        pub final_absolute_difference: String,
        pub final_relative_difference: String,
        /// Finest resolution attempted under the bounded policy. This is not
        /// described as accepted because `tolerance_met` can remain false
        /// after the maximum 4Q continuation.
        pub final_resolution: usize,
        pub tolerance_met: bool,
    }

    /// First-class evidence that a retained target-distance grid resolves the
    /// represented coefficient state under a fixed, versioned policy.
    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct PortableDistanceResolutionEvidence {
        pub schema_version: u32,
        /// SHA-256 of the canonical private runtime target specification.
        pub target_definition_digest: String,
        pub lambda_squared: String,
        pub n_modes: usize,
        pub precision_bits: u32,
        pub alpha: String,
        pub normalization: String,
        pub coefficient_count: usize,
        pub coefficient_tail: Vec<PortableCoefficientTailEvidence>,
        pub refinement_factor: usize,
        pub maximum_refinement_multiplier: usize,
        pub relative_tolerance: String,
        pub relative_difference_denominator: String,
        pub zero_denominator_fallback: String,
        /// Uniform-grid rules are refined in their original scheme and grid
        /// variable. Gauss--Legendre remains the independent-family
        /// cross-check retained by `ccm_target_distance` and is not doubled.
        pub refinements: Vec<PortableRuleResolutionEvidence>,
    }

    /// One sampled bracket in which the target residual changes sign.
    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct PortableTargetResidualCrossingBracket {
        pub left_u: String,
        pub right_u: String,
        pub left_residual: String,
        pub right_residual: String,
    }

    /// Signed and one-sided residual masses under one retained distance rule.
    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct PortableRuleTargetResidualAnalysis {
        pub rule_family: String,
        pub quadrature_rule: String,
        pub grid_variable: String,
        pub resolution: usize,
        /// Existing absolute target-residual measurement.
        pub absolute_residual_mass: String,
        /// Signed target-residual mass under the same rule.
        pub signed_residual_mass: String,
        /// `(absolute_residual_mass + signed_residual_mass) / 2`.
        pub positive_residual_mass: String,
        /// `(absolute_residual_mass - signed_residual_mass) / 2`.
        pub negative_residual_mass: String,
    }

    /// First-class diagnostic describing the sign structure hidden by the
    /// absolute value in a target-distance measurement.
    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct PortableTargetResidualAnalysis {
        pub schema_version: u32,
        /// SHA-256 of the canonical private runtime target specification.
        pub target_definition_digest: String,
        pub lambda_squared: String,
        pub n_modes: usize,
        pub precision_bits: u32,
        pub alpha: String,
        pub normalization: String,
        /// The already-retained profile grid on which pointwise diagnostics
        /// and crossing brackets are evaluated.
        pub sampling_grid_variable: String,
        pub sample_count: usize,
        /// Sign of the target residual at every retained profile abscissa: -1, 0, or 1.
        pub sample_signs: Vec<i8>,
        pub crossing_bracket_policy: String,
        pub crossing_brackets: Vec<PortableTargetResidualCrossingBracket>,
        pub maximum_sampled_residual: String,
        pub maximum_sampled_residual_u: String,
        pub minimum_sampled_residual: String,
        pub minimum_sampled_residual_u: String,
        /// Versioned handling for a sub-precision `|signed| > absolute`
        /// discrepancy caused by replaying retained decimal coefficients.
        pub mass_consistency_policy: String,
        pub one_sided_mass_derivation: String,
        /// One entry per target-distance rule, in its original order.
        pub measurements: Vec<PortableRuleTargetResidualAnalysis>,
    }

    /// Render at the decimal width the binary precision actually carries.
    /// The projection integrals use the profile's own grid, not an independent
    /// quadrature, so the decomposition inherits the profile's sampling.
    const DEVIATION_QUADRATURE_RULE: &str = "trapezoid_on_retained_profile_grid";
    const DEVIATION_SIGN_CONVENTION: &str =
        "sign follows the runtime-supplied auxiliary profile without reorientation";

    /// One projection of a profile deviation onto the auxiliary profile.
    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct PortableMetricDeviationProjection {
        /// Metric identifier. Every number in this entry was produced under it;
        /// amplitudes from different metrics are not comparable.
        pub metric: String,
        /// Signed least-squares amplitude; zero at a crossing.
        pub amplitude: String,
        pub deviation_norm: String,
        pub reference_norm: String,
        pub residual_norm: String,
        pub relative_residual: String,
    }

    /// Decomposition against the runtime-supplied auxiliary profile.
    ///
    /// Both readings of the distance functional's `u^(-1/2)` weight are
    /// retained, because they are not equivalent and an amplitude without its
    /// metric is not a recoverable number. No law relating amplitudes across
    /// configurations is computed or implied here.
    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct PortableDeviationDecomposition {
        /// Always 3. An interim pre-release build of 0.14.1 serialized a
        /// draft field name in this entry's projections under schema 2; the
        /// final schema is 3 so no interim payload can be accepted under the
        /// final identity.
        pub schema_version: u32,
        /// SHA-256 of the canonical private runtime target specification.
        pub target_definition_digest: String,
        pub lambda_squared: String,
        pub n_modes: usize,
        pub precision_bits: u32,
        pub normalization: String,
        pub sampling_grid_variable: String,
        pub sample_count: usize,
        /// Solved parameter carried by the auxiliary profile, at this precision.
        pub auxiliary_parameter: String,
        pub quadrature_rule: String,
        pub sign_convention: String,
        /// One entry per metric, in a fixed order; both are always retained.
        pub projections: Vec<PortableMetricDeviationProjection>,
    }

    /// Decompose a retained profile against the auxiliary profile.
    ///
    /// Depends only on the profile artifact, so it backfills onto configurations
    /// captured before this artifact existed without repeating an eigensolve.
    /// Structural acceptance for a retained deviation decomposition.
    ///
    /// Binds every identity field to the request the way the profile and
    /// distance validators do, so a payload cannot be accepted under a key it
    /// does not belong to. Deliberately does not recompute the projection: a
    /// cache hit must not evaluate either profile and both inner products just to
    /// be accepted.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn structural_deviation_decomposition_check(
        artifact: &PortableDeviationDecomposition,
        lambda_squared: &str,
        n_modes: usize,
        precision_bits: u32,
        variable: GridVariable,
        expected_samples: usize,
        target_definition_digest: &str,
    ) -> std::result::Result<(), xc_cache::CacheError> {
        if artifact.schema_version != 3
            || artifact.target_definition_digest != target_definition_digest
            || artifact.lambda_squared != lambda_squared
            || artifact.n_modes != n_modes
            || artifact.precision_bits != precision_bits
            || artifact.sampling_grid_variable != variable.as_str()
            || artifact.sample_count != expected_samples
            || artifact.normalization != "f(1)=1"
            || artifact.quadrature_rule != DEVIATION_QUADRATURE_RULE
            || artifact.sign_convention != DEVIATION_SIGN_CONVENTION
        {
            return Err(invalid_retained_payload(
                "CCM deviation decomposition does not match its request",
            ));
        }
        if artifact.projections.len() != 2 {
            return Err(invalid_retained_payload(
                "CCM deviation decomposition must retain both metrics",
            ));
        }
        let expected_metrics = [
            crate::deviation::DeviationMetric::FactorWeighted.as_str(),
            crate::deviation::DeviationMetric::IntegrandWeighted.as_str(),
        ];
        for (projection, expected) in artifact.projections.iter().zip(expected_metrics) {
            if projection.metric != expected {
                return Err(invalid_retained_payload(
                    "CCM deviation decomposition retained an unexpected metric",
                ));
            }
            // The amplitude is signed and passes through zero at a crossing;
            // the norms and the relative residual cannot be negative.
            parse_retained_float(&projection.amplitude, precision_bits, "amplitude")?;
            for (field, text) in [
                ("deviation_norm", &projection.deviation_norm),
                ("residual_norm", &projection.residual_norm),
                ("relative_residual", &projection.relative_residual),
            ] {
                let value = parse_retained_float(text, precision_bits, field)?;
                if value < 0u32 {
                    return Err(invalid_retained_payload(format!(
                        "CCM deviation decomposition {field} is negative"
                    )));
                }
            }
            let reference_norm =
                parse_retained_float(&projection.reference_norm, precision_bits, "reference_norm")?;
            if reference_norm <= 0u32 {
                return Err(invalid_retained_payload(
                    "CCM deviation decomposition reference norm must be positive",
                ));
            }
        }
        parse_retained_float(
            &artifact.auxiliary_parameter,
            precision_bits,
            "auxiliary_parameter",
        )?;
        Ok(())
    }

    pub(crate) fn compute_deviation_decomposition_payload(
        profile: &PortableEigenfunctionProfile,
        prec: u32,
    ) -> Result<PortableDeviationDecomposition> {
        use crate::deviation::hp::project;
        use crate::deviation::DeviationMetric;

        if profile.u_values.len() != profile.f_values.len() {
            anyhow::bail!(
                "retained profile has {} abscissae and {} values",
                profile.u_values.len(),
                profile.f_values.len()
            );
        }
        if profile.u_values.len() < 2 {
            anyhow::bail!("deviation decomposition requires at least two profile samples");
        }

        let parse = |text: &str, field: &str| -> Result<Float> {
            let parsed = Float::parse(text)
                .map_err(|error| anyhow::anyhow!("invalid retained {field}: {error}"))?;
            let value = Float::with_val(prec, parsed);
            if !value.is_finite() {
                anyhow::bail!("retained {field} must be finite");
            }
            Ok(value)
        };

        let working = prec.saturating_add(64);
        let target = crate::target::hp::TargetEvaluator::from_environment(working)?;
        let parameter = target
            .auxiliary_parameter()
            .ok_or_else(|| anyhow::anyhow!("target specification has no auxiliary profile"))?;

        let mut us = Vec::with_capacity(profile.u_values.len());
        let mut deviation = Vec::with_capacity(profile.u_values.len());
        let mut reference = Vec::with_capacity(profile.u_values.len());
        for (index, (u_text, f_text)) in profile.u_values.iter().zip(&profile.f_values).enumerate()
        {
            let u = parse(u_text, &format!("profile abscissa {index}"))?;
            let f = parse(f_text, &format!("profile value {index}"))?;
            let mut d = f;
            d -= target.value(&u);
            reference.push(target.auxiliary_value(&u)?);
            deviation.push(d);
            us.push(u);
        }

        let mut projections = Vec::with_capacity(2);
        for metric in [
            DeviationMetric::FactorWeighted,
            DeviationMetric::IntegrandWeighted,
        ] {
            let projected = project(&us, &deviation, &reference, metric, prec)?;
            projections.push(PortableMetricDeviationProjection {
                metric: metric.as_str().to_owned(),
                amplitude: decimal(&projected.amplitude, prec),
                deviation_norm: decimal(&projected.deviation_norm, prec),
                reference_norm: decimal(&projected.reference_norm, prec),
                residual_norm: decimal(&projected.residual_norm, prec),
                relative_residual: decimal(&projected.relative_residual, prec),
            });
        }

        Ok(PortableDeviationDecomposition {
            schema_version: 3,
            target_definition_digest: target.definition_digest().to_owned(),
            lambda_squared: profile.lambda_squared.clone(),
            n_modes: profile.n_modes,
            precision_bits: prec,
            normalization: profile.normalization.clone(),
            sampling_grid_variable: profile.grid_variable.clone(),
            sample_count: profile.u_values.len(),
            auxiliary_parameter: decimal(&parameter, prec),
            quadrature_rule: DEVIATION_QUADRATURE_RULE.to_owned(),
            sign_convention: DEVIATION_SIGN_CONVENTION.to_owned(),
            projections,
        })
    }

    pub(crate) fn decimal(value: &Float, prec: u32) -> String {
        let digits = ((f64::from(prec) * std::f64::consts::LOG10_2) as usize).max(20);
        value.to_string_radix(10, Some(digits))
    }

    fn lambda_squared_identity(params: &crate::ccm::CcmParams) -> String {
        if params.lambda_sq.is_integer {
            params.lambda_sq.value_u64.to_string()
        } else {
            format!("{:?}", params.lambda_sq.value_f64)
        }
    }

    /// Sample the eigenfunction on the profile grid used for a measurement.
    ///
    /// The profile uses the selected grid variable, but `steps` is an
    /// independent display/reconstruction resolution. It is not necessarily
    /// the node set used by any captured distance rule (in particular for a
    /// Gauss--Legendre rule or when `profile_steps` differs from the rule's
    /// resolution).
    fn sample_profile(
        eigenfunction: &WeilEigenfunction,
        lambda: &Float,
        steps: usize,
        variable: GridVariable,
        prec: u32,
    ) -> (Vec<Float>, Vec<Float>) {
        let working = prec.saturating_add(GUARD_BITS);
        let one = Float::with_val(working, 1u32);
        let (lo, hi) = match variable {
            GridVariable::U => (one.clone(), Float::with_val(working, lambda)),
            GridVariable::LogU => (
                Float::with_val(working, 0u32),
                Float::with_val(working, lambda).ln(),
            ),
        };
        let mut step = Float::with_val(working, &hi - &lo);
        step /= steps as u32;
        let mut u_values = Vec::with_capacity(steps + 1);
        let mut f_values = Vec::with_capacity(steps + 1);
        for index in 0..=steps {
            let mut point = step.clone();
            point *= index as u32;
            point += &lo;
            let u = match variable {
                GridVariable::U => point,
                GridVariable::LogU => point.exp(),
            };
            f_values.push(Float::with_val(prec, eigenfunction.eval(&u)));
            u_values.push(Float::with_val(prec, u));
        }
        (u_values, f_values)
    }

    fn invalid_retained_payload(detail: impl Into<String>) -> xc_cache::CacheError {
        xc_cache::CacheError::InvalidManifest(detail.into())
    }

    fn parse_retained_float(
        text: &str,
        precision_bits: u32,
        field: &str,
    ) -> std::result::Result<Float, xc_cache::CacheError> {
        let parsed = Float::parse(text).map_err(|error| {
            invalid_retained_payload(format!("invalid retained {field} decimal: {error}"))
        })?;
        let value = Float::with_val(precision_bits, parsed);
        if !value.is_finite() {
            return Err(invalid_retained_payload(format!(
                "retained {field} must be finite"
            )));
        }
        Ok(value)
    }

    pub(crate) fn validate_portable_eigenfunction_profile(
        artifact: &PortableEigenfunctionProfile,
        lambda_squared: &str,
        n_modes: usize,
        precision_bits: u32,
        variable: GridVariable,
        profile_steps: usize,
    ) -> std::result::Result<(), xc_cache::CacheError> {
        let expected_samples = profile_steps.checked_add(1).ok_or_else(|| {
            invalid_retained_payload("CCM eigenfunction profile sample count overflows usize")
        })?;
        let expected_coefficients = n_modes.checked_add(1).ok_or_else(|| {
            invalid_retained_payload("CCM eigenfunction coefficient count overflows usize")
        })?;
        if artifact.schema_version != 1
            || artifact.lambda_squared != lambda_squared
            || artifact.n_modes != n_modes
            || artifact.precision_bits != precision_bits
            || artifact.grid_variable != variable.as_str()
            || artifact.sample_count != expected_samples
            || artifact.normalization != "f(1)=1"
            || artifact.u_values.len() != expected_samples
            || artifact.f_values.len() != expected_samples
            || artifact.normalized_coefficients.len() != expected_coefficients
        {
            return Err(invalid_retained_payload(
                "CCM eigenfunction profile does not match its request",
            ));
        }

        let mut previous_u: Option<Float> = None;
        for value in &artifact.u_values {
            let u = parse_retained_float(value, precision_bits, "profile abscissa")?;
            if u <= 0u32 || previous_u.as_ref().is_some_and(|previous| &u <= previous) {
                return Err(invalid_retained_payload(
                    "CCM eigenfunction profile abscissae must be positive and strictly ascending",
                ));
            }
            previous_u = Some(u);
        }
        for value in &artifact.f_values {
            parse_retained_float(value, precision_bits, "profile value")?;
        }
        for value in &artifact.normalized_coefficients {
            parse_retained_float(value, precision_bits, "normalized coefficient")?;
        }
        Ok(())
    }

    pub(crate) struct TargetDistanceValidationRequest<'a> {
        pub(crate) target_definition_digest: &'a str,
        pub(crate) lambda_squared: &'a str,
        pub(crate) n_modes: usize,
        pub(crate) precision_bits: u32,
        pub(crate) alpha: &'a Float,
        pub(crate) rules: &'a [WeightedIntegrationRule],
        pub(crate) expected_eigenvalue: &'a Float,
    }

    pub(crate) fn validate_portable_target_distance(
        artifact: &PortableTargetDistance,
        request: TargetDistanceValidationRequest<'_>,
    ) -> std::result::Result<(), xc_cache::CacheError> {
        let TargetDistanceValidationRequest {
            target_definition_digest,
            lambda_squared,
            n_modes,
            precision_bits,
            alpha,
            rules,
            expected_eigenvalue,
        } = request;
        if artifact.schema_version != 2
            || artifact.target_definition_digest != target_definition_digest
            || artifact.lambda_squared != lambda_squared
            || artifact.n_modes != n_modes
            || artifact.precision_bits != precision_bits
            || artifact.alpha != decimal(alpha, precision_bits)
            || artifact.eigenvalue != decimal(expected_eigenvalue, precision_bits)
            || artifact.measurements.len() != rules.len()
        {
            return Err(invalid_retained_payload(
                "CCM target distance does not match its request",
            ));
        }
        for (entry, rule) in artifact.measurements.iter().zip(rules) {
            if entry.rule_family != rule.family()
                || entry.quadrature_rule != rule.rule()
                || entry.grid_variable != rule.variable().as_str()
                || entry.resolution != rule.resolution()
            {
                return Err(invalid_retained_payload(
                    "CCM target-distance rule does not match its request",
                ));
            }
            let distance =
                parse_retained_float(&entry.distance_to_target, precision_bits, "target distance")?;
            let norm = parse_retained_float(
                &entry.eigenfunction_norm,
                precision_bits,
                "eigenfunction norm",
            )?;
            if distance < 0u32 || norm < 0u32 {
                return Err(invalid_retained_payload(
                    "retained distances and norms must be nonnegative",
                ));
            }
        }
        parse_retained_float(&artifact.eigenvalue, precision_bits, "eigenvalue")?;
        Ok(())
    }

    fn absolute_and_relative_difference(
        coarser: &Float,
        finer: &Float,
        precision_bits: u32,
    ) -> (Float, Float) {
        let working = precision_bits.saturating_add(GUARD_BITS);
        let absolute = Float::with_val(working, coarser - finer).abs();
        let denominator = Float::with_val(working, finer).abs();
        let relative = if denominator == 0u32 {
            absolute.clone()
        } else {
            Float::with_val(working, &absolute / denominator)
        };
        (
            Float::with_val(precision_bits, absolute),
            Float::with_val(precision_bits, relative),
        )
    }

    /// Grid guard bits used by `xc_numerics::grid_integral::hp`.
    ///
    /// Mirrored here so the abscissae generated for the precomputed table are
    /// bit-identical to the ones the integrator will ask for. A mismatch is
    /// not a correctness problem - a miss falls back to a direct evaluation -
    /// but it would silently forfeit the reuse.
    const GRID_GUARD_BITS: u32 = 32;

    /// Every abscissa a uniform rule will pass to its integrand, in order.
    ///
    /// Reproduces `xc_numerics::grid_integral::hp::uniform_sum`: the same
    /// working precision, the same `lo + h*i` construction, the same offsets
    /// per scheme, and the same `exp` for a log-variable grid.
    fn uniform_rule_abscissae(
        lambda: &Float,
        scheme: UniformGridScheme,
        variable: GridVariable,
        steps: usize,
        prec: u32,
    ) -> Vec<Float> {
        let working = prec.saturating_add(GRID_GUARD_BITS);
        let (lo, hi) = match variable {
            GridVariable::U => (
                Float::with_val(working, 1u32),
                Float::with_val(working, lambda),
            ),
            GridVariable::LogU => (
                Float::with_val(working, 1u32).ln(),
                Float::with_val(working, lambda).ln(),
            ),
        };
        let mut h = Float::with_val(working, &hi - &lo);
        h /= steps as u32;
        let point = |i: f64| {
            let mut x = h.clone();
            x *= Float::with_val(working, i);
            x += &lo;
            x
        };
        let mut offsets: Vec<f64> = match scheme {
            UniformGridScheme::LeftRiemann => (0..steps).map(|i| i as f64).collect(),
            UniformGridScheme::RightRiemann => (1..=steps).map(|i| i as f64).collect(),
            UniformGridScheme::Midpoint => (0..steps).map(|i| i as f64 + 0.5).collect(),
            UniformGridScheme::Trapezoid => (1..steps).map(|i| i as f64).collect(),
        };
        let mut points: Vec<Float> = Vec::with_capacity(offsets.len() + 2);
        if scheme == UniformGridScheme::Trapezoid {
            points.push(Float::with_val(working, &lo));
            points.push(Float::with_val(working, &hi));
        }
        points.extend(offsets.drain(..).map(point));
        match variable {
            GridVariable::U => points,
            GridVariable::LogU => points.into_iter().map(|t| t.exp()).collect(),
        }
    }

    /// Eigenfunction values for a known set of abscissae, evaluated once.
    ///
    /// Replaces a lazily-populated memoization cache. The abscissae a uniform
    /// refinement will visit are known in advance, so the whole set is
    /// deduplicated on MPFR's exact integer/exponent key and evaluated in one
    /// parallel pass, then frozen. Two things follow. The 50% reuse that grid
    /// nesting provides is kept - a `2Q` grid contains every `Q` abscissa
    /// bit-exactly, because `h_2Q = h_Q/2` is exact and `h_2Q*2i` and `h_Q*i`
    /// round identically. And because nothing mutates during integration, the
    /// lookup is a pure function: no interior mutability, so it is `Sync` and
    /// safe to call from a parallel quadrature.
    ///
    /// Quadrature accumulation order is unaffected, and a miss falls back to a
    /// direct evaluation, so retained values are bit-identical either way.
    pub(crate) struct PrecomputedEigenfunctionValues<'a> {
        eigenfunction: &'a WeilEigenfunction,
        values: std::collections::HashMap<(Integer, i32), Float>,
    }

    impl<'a> PrecomputedEigenfunctionValues<'a> {
        /// Seed from an earlier pass, then evaluate every abscissa in `rules`
        /// that is not already known.
        pub(crate) fn build(
            eigenfunction: &'a WeilEigenfunction,
            seeds: Option<&[(Float, Float)]>,
            rules: &[(UniformGridScheme, GridVariable, usize)],
            lambda: &Float,
            prec: u32,
        ) -> Self {
            use rayon::prelude::*;

            let mut values: std::collections::HashMap<(Integer, i32), Float> =
                std::collections::HashMap::new();
            if let Some(seeds) = seeds {
                values.reserve(seeds.len());
                for (u, value) in seeds {
                    if let Some(key) = u.to_integer_exp() {
                        values.insert(key, value.clone());
                    }
                }
            }

            // Deduplicate across every requested rule before evaluating.
            let mut pending: Vec<Float> = Vec::new();
            let mut seen: std::collections::HashSet<(Integer, i32)> =
                values.keys().cloned().collect();
            for (scheme, variable, steps) in rules.iter().copied() {
                for u in uniform_rule_abscissae(lambda, scheme, variable, steps, prec) {
                    let Some(key) = u.to_integer_exp() else {
                        continue;
                    };
                    if seen.insert(key) {
                        pending.push(u);
                    }
                }
            }

            let evaluated: Vec<(Float, Float)> = pending
                .par_iter()
                .map(|u| (u.clone(), eigenfunction.eval(u)))
                .collect();
            values.reserve(evaluated.len());
            for (u, value) in evaluated {
                if let Some(key) = u.to_integer_exp() {
                    values.insert(key, value);
                }
            }

            Self {
                eigenfunction,
                values,
            }
        }

        /// A hit returns the retained value; a miss evaluates directly. Pure.
        pub(crate) fn eval(&self, u: &Float) -> Float {
            match u.to_integer_exp() {
                Some(key) => match self.values.get(&key) {
                    Some(value) => value.clone(),
                    None => self.eigenfunction.eval(u),
                },
                None => self.eigenfunction.eval(u),
            }
        }

        /// Number of abscissae retained. Used by tests to pin the reuse rate.
        #[cfg(test)]
        pub(crate) fn retained(&self) -> usize {
            self.values.len()
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn compute_distance_resolution_evidence(
        eigenfunction: &WeilEigenfunction,
        lambda: &Float,
        alpha: &Float,
        rules: &[WeightedIntegrationRule],
        base_distances: &[WeightedGridValueHp],
        target_definition_digest: &str,
        lambda_squared: &str,
        precision_bits: u32,
    ) -> Result<PortableDistanceResolutionEvidence> {
        compute_distance_resolution_evidence_with_samples(
            eigenfunction,
            lambda,
            alpha,
            rules,
            ResolutionEvidenceEvaluationSource {
                base_distances,
                base_value_samples: None,
            },
            target_definition_digest,
            lambda_squared,
            precision_bits,
        )
    }

    pub(crate) struct ResolutionEvidenceEvaluationSource<'a> {
        pub(crate) base_distances: &'a [WeightedGridValueHp],
        pub(crate) base_value_samples: Option<&'a [Vec<(Float, Float)>]>,
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn compute_distance_resolution_evidence_with_samples(
        eigenfunction: &WeilEigenfunction,
        lambda: &Float,
        alpha: &Float,
        rules: &[WeightedIntegrationRule],
        evaluation_source: ResolutionEvidenceEvaluationSource<'_>,
        target_definition_digest: &str,
        lambda_squared: &str,
        precision_bits: u32,
    ) -> Result<PortableDistanceResolutionEvidence> {
        if rules.len() != evaluation_source.base_distances.len() {
            anyhow::bail!(
                "resolution evidence needs one base distance per rule ({} rules, {} distances)",
                rules.len(),
                evaluation_source.base_distances.len()
            );
        }
        if evaluation_source
            .base_value_samples
            .is_some_and(|samples| samples.len() != rules.len())
        {
            anyhow::bail!("resolution evidence has the wrong base-sample count");
        }
        let working = precision_bits.saturating_add(GUARD_BITS);
        let coefficients = eigenfunction
            .normalized_coefficients()
            .into_iter()
            .map(|coefficient| Float::with_val(precision_bits, coefficient))
            .collect::<Vec<_>>();
        let n_modes = coefficients.len() - 1;
        let mut coefficient_tail = Vec::with_capacity(RESOLUTION_EVIDENCE_THRESHOLD_DECADES.len());
        for threshold_decades in RESOLUTION_EVIDENCE_THRESHOLD_DECADES {
            let threshold_text = format!("1e-{threshold_decades}");
            let parsed = Float::parse(&threshold_text)
                .map_err(|error| anyhow::anyhow!("invalid resolution threshold: {error}"))?;
            let threshold = Float::with_val(precision_bits, parsed);
            let effective_bandwidth = coefficients
                .iter()
                .rposition(|coefficient| coefficient.clone().abs() > threshold);
            let discarded_start = effective_bandwidth.map_or(0, |index| index + 1);
            let mut one_sided_l1 = Float::with_val(working, 0u32);
            let mut pointwise_bound = Float::with_val(working, 0u32);
            let mut cosine_l2_squared = Float::with_val(working, 0u32);
            for (index, coefficient) in coefficients.iter().enumerate().skip(discarded_start) {
                let magnitude = coefficient.clone().abs();
                one_sided_l1 += &magnitude;
                let mut pointwise_term = magnitude;
                if index > 0 {
                    pointwise_term *= 2u32;
                }
                pointwise_bound += pointwise_term;

                let mut squared = Float::with_val(working, coefficient * coefficient);
                if index > 0 {
                    squared *= 2u32;
                }
                cosine_l2_squared += squared;
            }
            coefficient_tail.push(PortableCoefficientTailEvidence {
                threshold: threshold_text,
                effective_bandwidth,
                discarded_one_sided_l1: decimal(&one_sided_l1, precision_bits),
                discarded_cosine_pointwise_bound: decimal(&pointwise_bound, precision_bits),
                discarded_cosine_l2: decimal(&cosine_l2_squared.sqrt(), precision_bits),
            });
        }

        let tolerance_parsed = Float::parse(RESOLUTION_EVIDENCE_RELATIVE_TOLERANCE)
            .map_err(|error| anyhow::anyhow!("invalid resolution tolerance: {error}"))?;
        let tolerance = Float::with_val(precision_bits, tolerance_parsed);
        let mut refinements = Vec::new();
        for (rule_index, (rule, base_distance)) in rules
            .iter()
            .zip(evaluation_source.base_distances)
            .enumerate()
        {
            let WeightedIntegrationRule::UniformGrid {
                scheme,
                variable,
                steps,
            } = *rule
            else {
                continue;
            };
            let twice_resolution = steps
                .checked_mul(RESOLUTION_EVIDENCE_REFINEMENT_FACTOR)
                .ok_or_else(|| anyhow::anyhow!("2Q resolution overflows usize for Q = {steps}"))?;
            let twice_rule = WeightedIntegrationRule::UniformGrid {
                scheme,
                variable,
                steps: twice_resolution,
            };
            let samples = evaluation_source
                .base_value_samples
                .and_then(|all_samples| all_samples.get(rule_index))
                .filter(|samples| !samples.is_empty());
            // Both refinement levels are enumerated now so their shared
            // abscissae are evaluated once, in one parallel pass, rather than
            // lazily during integration. 4Q is included even though it is
            // conditional: it shares every 2Q abscissa, so enumerating it
            // early costs one extra evaluation set only when 4Q actually runs,
            // and lets the whole ladder be evaluated in a single pass.
            let four_resolution_hint = steps.checked_mul(RESOLUTION_EVIDENCE_MAXIMUM_MULTIPLIER);
            let mut precompute_rules = vec![(scheme, variable, twice_resolution)];
            if let Some(four) = four_resolution_hint {
                precompute_rules.push((scheme, variable, four));
            }
            let exact_values = (scheme != UniformGridScheme::Midpoint).then(|| {
                PrecomputedEigenfunctionValues::build(
                    eigenfunction,
                    samples.map(Vec::as_slice),
                    &precompute_rules,
                    lambda,
                    precision_bits,
                )
            });
            let twice_distance = distance_to_target_with_tables_bound(
                |u: &Float| {
                    exact_values
                        .as_ref()
                        .map_or_else(|| eigenfunction.eval(u), |values| values.eval(u))
                },
                lambda,
                alpha,
                twice_rule,
                precision_bits,
                None,
                Some(target_definition_digest),
            )?;
            let (q_to_2q_absolute, q_to_2q_relative) = absolute_and_relative_difference(
                &base_distance.value,
                &twice_distance.value,
                precision_bits,
            );
            let q_to_2q_absolute_text = decimal(&q_to_2q_absolute, precision_bits);
            let q_to_2q_relative_text = decimal(&q_to_2q_relative, precision_bits);

            let (
                four_times_resolution,
                four_times_distance,
                final_absolute,
                final_relative,
                final_resolution,
            ) = if q_to_2q_relative <= tolerance {
                (
                    None,
                    None,
                    q_to_2q_absolute,
                    q_to_2q_relative,
                    twice_resolution,
                )
            } else {
                let four_resolution = steps
                    .checked_mul(RESOLUTION_EVIDENCE_MAXIMUM_MULTIPLIER)
                    .ok_or_else(|| {
                        anyhow::anyhow!("4Q resolution overflows usize for Q = {steps}")
                    })?;
                let four_rule = WeightedIntegrationRule::UniformGrid {
                    scheme,
                    variable,
                    steps: four_resolution,
                };
                let four_distance = distance_to_target_with_tables_bound(
                    |u: &Float| {
                        exact_values
                            .as_ref()
                            .map_or_else(|| eigenfunction.eval(u), |values| values.eval(u))
                    },
                    lambda,
                    alpha,
                    four_rule,
                    precision_bits,
                    None,
                    Some(target_definition_digest),
                )?;
                let (absolute, relative) = absolute_and_relative_difference(
                    &twice_distance.value,
                    &four_distance.value,
                    precision_bits,
                );
                (
                    Some(four_resolution),
                    Some(decimal(&four_distance.value, precision_bits)),
                    absolute,
                    relative,
                    four_resolution,
                )
            };
            let tolerance_met = final_relative <= tolerance;
            refinements.push(PortableRuleResolutionEvidence {
                rule_family: rule.family().to_owned(),
                quadrature_rule: rule.rule().to_owned(),
                grid_variable: variable.as_str().to_owned(),
                base_resolution: steps,
                base_distance: decimal(&base_distance.value, precision_bits),
                twice_resolution,
                twice_distance: decimal(&twice_distance.value, precision_bits),
                q_to_2q_absolute_difference: q_to_2q_absolute_text,
                q_to_2q_relative_difference: q_to_2q_relative_text,
                four_times_resolution,
                four_times_distance,
                final_absolute_difference: decimal(&final_absolute, precision_bits),
                final_relative_difference: decimal(&final_relative, precision_bits),
                final_resolution,
                tolerance_met,
            });
        }
        if refinements.is_empty() {
            anyhow::bail!("resolution evidence requires at least one uniform-grid rule");
        }

        Ok(PortableDistanceResolutionEvidence {
            schema_version: 2,
            target_definition_digest: target_definition_digest.to_owned(),
            lambda_squared: lambda_squared.to_owned(),
            n_modes,
            precision_bits,
            alpha: decimal(alpha, precision_bits),
            normalization: "f(1)=1".to_owned(),
            coefficient_count: coefficients.len(),
            coefficient_tail,
            refinement_factor: RESOLUTION_EVIDENCE_REFINEMENT_FACTOR,
            maximum_refinement_multiplier: RESOLUTION_EVIDENCE_MAXIMUM_MULTIPLIER,
            relative_tolerance: RESOLUTION_EVIDENCE_RELATIVE_TOLERANCE.to_owned(),
            relative_difference_denominator: "absolute_finer_distance".to_owned(),
            zero_denominator_fallback: "absolute_difference".to_owned(),
            refinements,
        })
    }

    pub(crate) fn validate_portable_distance_resolution_evidence(
        artifact: &PortableDistanceResolutionEvidence,
        target_definition_digest: &str,
        lambda_squared: &str,
        n_modes: usize,
        precision_bits: u32,
        alpha: &Float,
        rules: &[WeightedIntegrationRule],
    ) -> std::result::Result<(), xc_cache::CacheError> {
        let expected_coefficients = n_modes.checked_add(1).ok_or_else(|| {
            invalid_retained_payload("CCM resolution-evidence coefficient count overflows usize")
        })?;
        let uniform_rules = rules
            .iter()
            .filter(|rule| matches!(rule, WeightedIntegrationRule::UniformGrid { .. }))
            .collect::<Vec<_>>();
        if uniform_rules.is_empty()
            || artifact.schema_version != 2
            || artifact.target_definition_digest != target_definition_digest
            || artifact.lambda_squared != lambda_squared
            || artifact.n_modes != n_modes
            || artifact.precision_bits != precision_bits
            || artifact.alpha != decimal(alpha, precision_bits)
            || artifact.normalization != "f(1)=1"
            || artifact.coefficient_count != expected_coefficients
            || artifact.coefficient_tail.len() != RESOLUTION_EVIDENCE_THRESHOLD_DECADES.len()
            || artifact.refinement_factor != RESOLUTION_EVIDENCE_REFINEMENT_FACTOR
            || artifact.maximum_refinement_multiplier != RESOLUTION_EVIDENCE_MAXIMUM_MULTIPLIER
            || artifact.relative_tolerance != RESOLUTION_EVIDENCE_RELATIVE_TOLERANCE
            || artifact.relative_difference_denominator != "absolute_finer_distance"
            || artifact.zero_denominator_fallback != "absolute_difference"
            || artifact.refinements.len() != uniform_rules.len()
        {
            return Err(invalid_retained_payload(
                "CCM distance resolution evidence does not match its request",
            ));
        }

        for (tail, threshold_decades) in artifact
            .coefficient_tail
            .iter()
            .zip(RESOLUTION_EVIDENCE_THRESHOLD_DECADES)
        {
            if tail.threshold != format!("1e-{threshold_decades}")
                || tail
                    .effective_bandwidth
                    .is_some_and(|bandwidth| bandwidth > n_modes)
            {
                return Err(invalid_retained_payload(
                    "CCM coefficient-tail evidence does not match its policy",
                ));
            }
            for (value, field) in [
                (&tail.discarded_one_sided_l1, "discarded one-sided L1"),
                (
                    &tail.discarded_cosine_pointwise_bound,
                    "discarded cosine pointwise bound",
                ),
                (&tail.discarded_cosine_l2, "discarded cosine L2"),
            ] {
                if parse_retained_float(value, precision_bits, field)? < 0u32 {
                    return Err(invalid_retained_payload(format!(
                        "retained {field} must be nonnegative"
                    )));
                }
            }
        }

        let tolerance = parse_retained_float(
            RESOLUTION_EVIDENCE_RELATIVE_TOLERANCE,
            precision_bits,
            "resolution tolerance",
        )?;
        for (entry, rule) in artifact.refinements.iter().zip(uniform_rules) {
            let twice_resolution = rule
                .resolution()
                .checked_mul(RESOLUTION_EVIDENCE_REFINEMENT_FACTOR)
                .ok_or_else(|| invalid_retained_payload("2Q resolution overflows usize"))?;
            let four_resolution = rule
                .resolution()
                .checked_mul(RESOLUTION_EVIDENCE_MAXIMUM_MULTIPLIER)
                .ok_or_else(|| invalid_retained_payload("4Q resolution overflows usize"))?;
            if entry.rule_family != rule.family()
                || entry.quadrature_rule != rule.rule()
                || entry.grid_variable != rule.variable().as_str()
                || entry.base_resolution != rule.resolution()
                || entry.twice_resolution != twice_resolution
            {
                return Err(invalid_retained_payload(
                    "CCM resolution-evidence rule does not match its request",
                ));
            }
            let mut values = vec![
                (&entry.base_distance, "base distance"),
                (&entry.twice_distance, "twice-refined distance"),
                (
                    &entry.q_to_2q_absolute_difference,
                    "Q-to-2Q absolute difference",
                ),
                (
                    &entry.q_to_2q_relative_difference,
                    "Q-to-2Q relative difference",
                ),
                (
                    &entry.final_absolute_difference,
                    "final absolute difference",
                ),
                (
                    &entry.final_relative_difference,
                    "final relative difference",
                ),
            ];
            if let Some(value) = &entry.four_times_distance {
                values.push((value, "four-times-refined distance"));
            }
            for (value, field) in values {
                if parse_retained_float(value, precision_bits, field)? < 0u32 {
                    return Err(invalid_retained_payload(format!(
                        "retained {field} must be nonnegative"
                    )));
                }
            }

            let q_to_2q_relative = parse_retained_float(
                &entry.q_to_2q_relative_difference,
                precision_bits,
                "Q-to-2Q relative difference",
            )?;
            let continued = q_to_2q_relative > tolerance;
            match (
                continued,
                entry.four_times_resolution,
                entry.four_times_distance.as_ref(),
            ) {
                (false, None, None) => {
                    if entry.final_resolution != twice_resolution
                        || entry.final_absolute_difference != entry.q_to_2q_absolute_difference
                        || entry.final_relative_difference != entry.q_to_2q_relative_difference
                    {
                        return Err(invalid_retained_payload(
                            "CCM Q/2Q resolution evidence has inconsistent final fields",
                        ));
                    }
                }
                (true, Some(resolution), Some(_)) if resolution == four_resolution => {
                    if entry.final_resolution != four_resolution {
                        return Err(invalid_retained_payload(
                            "CCM Q/2Q/4Q resolution evidence has the wrong final resolution",
                        ));
                    }
                }
                _ => {
                    return Err(invalid_retained_payload(
                        "CCM resolution evidence did not follow the deterministic Q/2Q/4Q policy",
                    ));
                }
            }
            let final_relative = parse_retained_float(
                &entry.final_relative_difference,
                precision_bits,
                "final relative difference",
            )?;
            if entry.tolerance_met != (final_relative <= tolerance) {
                return Err(invalid_retained_payload(
                    "CCM resolution-evidence tolerance verdict is inconsistent",
                ));
            }
        }
        Ok(())
    }

    pub(crate) struct TargetResidualAnalysisSource<'a> {
        pub(crate) eigenfunction: &'a WeilEigenfunction,
        pub(crate) lambda: &'a Float,
        pub(crate) alpha: &'a Float,
        pub(crate) rules: &'a [WeightedIntegrationRule],
        pub(crate) base_distances: &'a [WeightedGridValueHp],
        pub(crate) u_values: &'a [Float],
        pub(crate) f_values: &'a [Float],
        pub(crate) precomputed_signed_residuals: Option<&'a [Float]>,
        pub(crate) target_definition_digest: &'a str,
        pub(crate) lambda_squared: &'a str,
        pub(crate) sampling_variable: GridVariable,
        pub(crate) precision_bits: u32,
    }

    pub(crate) fn compute_target_residual_analysis(
        source: TargetResidualAnalysisSource<'_>,
    ) -> Result<PortableTargetResidualAnalysis> {
        compute_target_residual_analysis_with_tables(source, None)
    }

    fn compute_target_residual_analysis_with_tables(
        source: TargetResidualAnalysisSource<'_>,
        supplied_gl_tables: Option<&mut SharedGlTables>,
    ) -> Result<PortableTargetResidualAnalysis> {
        if source.rules.len() != source.base_distances.len() {
            anyhow::bail!(
                "residual analysis needs one base distance per rule ({} rules, {} distances)",
                source.rules.len(),
                source.base_distances.len()
            );
        }
        if source.u_values.is_empty() || source.u_values.len() != source.f_values.len() {
            anyhow::bail!("residual analysis needs matching nonempty profile samples");
        }
        if source
            .precomputed_signed_residuals
            .is_some_and(|values| values.len() != source.rules.len())
        {
            anyhow::bail!("residual analysis has the wrong signed-residual count");
        }
        let working = source.precision_bits.saturating_add(GUARD_BITS);
        let target = crate::target::hp::TargetEvaluator::from_environment(working)?;
        if target.definition_digest() != source.target_definition_digest {
            anyhow::bail!(
                "runtime target specification changed after its semantic identity was fixed"
            );
        }
        let mut residuals = Vec::with_capacity(source.u_values.len());
        let mut sample_signs = Vec::with_capacity(source.u_values.len());
        for (u, f_value) in source.u_values.iter().zip(source.f_values) {
            let residual = Float::with_val(source.precision_bits, f_value - target.value(u));
            sample_signs.push(if residual > 0u32 {
                1
            } else if residual < 0u32 {
                -1
            } else {
                0
            });
            residuals.push(residual);
        }

        let mut maximum_index = 0usize;
        let mut minimum_index = 0usize;
        for index in 1..residuals.len() {
            if residuals[index] > residuals[maximum_index] {
                maximum_index = index;
            }
            if residuals[index] < residuals[minimum_index] {
                minimum_index = index;
            }
        }
        let mut crossing_brackets = Vec::new();
        let mut previous_nonzero: Option<usize> = None;
        for (index, sign) in sample_signs.iter().enumerate() {
            if *sign == 0 {
                continue;
            }
            if let Some(previous) = previous_nonzero {
                if sample_signs[previous] != *sign {
                    crossing_brackets.push(PortableTargetResidualCrossingBracket {
                        left_u: decimal(&source.u_values[previous], source.precision_bits),
                        right_u: decimal(&source.u_values[index], source.precision_bits),
                        left_residual: decimal(&residuals[previous], source.precision_bits),
                        right_residual: decimal(&residuals[index], source.precision_bits),
                    });
                }
            }
            previous_nonzero = Some(index);
        }

        let signed_residuals = match source.precomputed_signed_residuals {
            Some(values) => values.to_vec(),
            None => {
                let mut local_gl_tables = SharedGlTables::new();
                let gl_tables = supplied_gl_tables.unwrap_or(&mut local_gl_tables);
                let mut values = Vec::with_capacity(source.rules.len());
                for rule in source.rules {
                    values.push(signed_residual_to_target_with_tables(
                        |u: &Float| source.eigenfunction.eval(u),
                        source.lambda,
                        source.alpha,
                        *rule,
                        source.precision_bits,
                        Some(&mut *gl_tables),
                        Some(source.target_definition_digest),
                    )?);
                }
                values
            }
        };
        let mut measurements = Vec::with_capacity(source.rules.len());
        for ((rule, base_distance), signed) in source
            .rules
            .iter()
            .zip(source.base_distances)
            .zip(&signed_residuals)
        {
            let absolute = Float::with_val(source.precision_bits, &base_distance.value);
            let mut signed = Float::with_val(source.precision_bits, signed);
            let signed_magnitude = Float::with_val(source.precision_bits, &signed).abs();
            if signed_magnitude > absolute {
                let excess = Float::with_val(
                    source.precision_bits,
                    Float::with_val(working, &signed_magnitude - &absolute),
                );
                let mut scale = Float::with_val(working, 1u32);
                if absolute > scale {
                    scale = Float::with_val(working, &absolute);
                }
                if signed_magnitude > scale {
                    scale = Float::with_val(working, &signed_magnitude);
                }
                scale >>= source.precision_bits.saturating_sub(8).max(1);
                if excess > scale {
                    anyhow::bail!(
                        "signed residual exceeds absolute residual under rule {}: absolute={}, signed={}, excess={}, tolerance={}",
                        rule.rule(),
                        decimal(&absolute, source.precision_bits),
                        decimal(&signed, source.precision_bits),
                        decimal(&excess, source.precision_bits),
                        decimal(&Float::with_val(source.precision_bits, scale), source.precision_bits),
                    );
                }
                signed = if signed < 0u32 {
                    Float::with_val(source.precision_bits, -&absolute)
                } else {
                    Float::with_val(source.precision_bits, &absolute)
                };
            }
            // Derive the one-sided masses from the values *as retained*, not
            // from the wider working-precision originals.
            //
            // `decimal` emits `precision_bits * log10(2)` digits, which is two
            // short of an exact round trip, so `parse(decimal(x))` differs from
            // `x` in the low bits. The reader recomputes
            // `(absolute +/- signed)/2` from the parsed decimals and requires
            // the result to reproduce the retained strings exactly, so deriving
            // here from the unrounded values leaves a quadruple that cannot
            // validate. Rounding through the retained text first makes the
            // stored four values self-consistent under the reader's own
            // arithmetic.
            let absolute_text = decimal(&absolute, source.precision_bits);
            let signed_text = decimal(&signed, source.precision_bits);
            let retained = |text: &str, field: &str| -> Result<Float> {
                let parsed = Float::parse(text).map_err(|error| {
                    anyhow::anyhow!("retained {field} is not a valid decimal: {error}")
                })?;
                Ok(Float::with_val(source.precision_bits, parsed))
            };
            let absolute = retained(&absolute_text, "absolute residual mass")?;
            let signed = retained(&signed_text, "signed residual mass")?;

            let mut positive = Float::with_val(working, &absolute + &signed);
            positive /= 2u32;
            let positive = Float::with_val(source.precision_bits, positive);
            let mut negative = Float::with_val(working, &absolute - &signed);
            negative /= 2u32;
            let negative = Float::with_val(source.precision_bits, negative);
            debug_assert!(positive >= 0u32 && negative >= 0u32);
            measurements.push(PortableRuleTargetResidualAnalysis {
                rule_family: rule.family().to_owned(),
                quadrature_rule: rule.rule().to_owned(),
                grid_variable: rule.variable().as_str().to_owned(),
                resolution: rule.resolution(),
                absolute_residual_mass: absolute_text,
                signed_residual_mass: signed_text,
                positive_residual_mass: decimal(&positive, source.precision_bits),
                negative_residual_mass: decimal(&negative, source.precision_bits),
            });
        }

        Ok(PortableTargetResidualAnalysis {
            schema_version: 2,
            target_definition_digest: source.target_definition_digest.to_owned(),
            lambda_squared: source.lambda_squared.to_owned(),
            n_modes: source.eigenfunction.normalized_coefficients().len() - 1,
            precision_bits: source.precision_bits,
            alpha: decimal(source.alpha, source.precision_bits),
            normalization: "f(1)=1".to_owned(),
            sampling_grid_variable: source.sampling_variable.as_str().to_owned(),
            sample_count: source.u_values.len(),
            sample_signs,
            crossing_bracket_policy: "adjacent_nonzero_profile_samples_v1".to_owned(),
            crossing_brackets,
            maximum_sampled_residual: decimal(&residuals[maximum_index], source.precision_bits),
            maximum_sampled_residual_u: decimal(
                &source.u_values[maximum_index],
                source.precision_bits,
            ),
            minimum_sampled_residual: decimal(&residuals[minimum_index], source.precision_bits),
            minimum_sampled_residual_u: decimal(
                &source.u_values[minimum_index],
                source.precision_bits,
            ),
            mass_consistency_policy: RESIDUAL_MASS_CONSISTENCY_POLICY.to_owned(),
            one_sided_mass_derivation: "positive=(absolute+signed)/2;negative=(absolute-signed)/2"
                .to_owned(),
            measurements,
        })
    }

    pub(crate) struct TargetResidualAnalysisValidationRequest<'a> {
        pub(crate) target_definition_digest: &'a str,
        pub(crate) lambda_squared: &'a str,
        pub(crate) n_modes: usize,
        pub(crate) precision_bits: u32,
        pub(crate) alpha: &'a Float,
        pub(crate) rules: &'a [WeightedIntegrationRule],
        pub(crate) variable: GridVariable,
        pub(crate) profile_steps: usize,
    }

    pub(crate) fn validate_portable_target_residual_analysis(
        artifact: &PortableTargetResidualAnalysis,
        request: TargetResidualAnalysisValidationRequest<'_>,
    ) -> std::result::Result<(), xc_cache::CacheError> {
        let TargetResidualAnalysisValidationRequest {
            target_definition_digest,
            lambda_squared,
            n_modes,
            precision_bits,
            alpha,
            rules,
            variable,
            profile_steps,
        } = request;
        let expected_samples = profile_steps.checked_add(1).ok_or_else(|| {
            invalid_retained_payload("CCM residual-analysis sample count overflows usize")
        })?;
        if artifact.schema_version != 2
            || artifact.target_definition_digest != target_definition_digest
            || artifact.lambda_squared != lambda_squared
            || artifact.n_modes != n_modes
            || artifact.precision_bits != precision_bits
            || artifact.alpha != decimal(alpha, precision_bits)
            || artifact.normalization != "f(1)=1"
            || artifact.sampling_grid_variable != variable.as_str()
            || artifact.sample_count != expected_samples
            || artifact.sample_signs.len() != expected_samples
            || artifact
                .sample_signs
                .iter()
                .any(|sign| !matches!(sign, -1..=1))
            || artifact.crossing_bracket_policy != "adjacent_nonzero_profile_samples_v1"
            || artifact.mass_consistency_policy != RESIDUAL_MASS_CONSISTENCY_POLICY
            || artifact.one_sided_mass_derivation
                != "positive=(absolute+signed)/2;negative=(absolute-signed)/2"
            || artifact.measurements.len() != rules.len()
        {
            return Err(invalid_retained_payload(
                "CCM target residual analysis does not match its request",
            ));
        }
        let maximum = parse_retained_float(
            &artifact.maximum_sampled_residual,
            precision_bits,
            "maximum sampled residual",
        )?;
        let minimum = parse_retained_float(
            &artifact.minimum_sampled_residual,
            precision_bits,
            "minimum sampled residual",
        )?;
        if maximum < minimum {
            return Err(invalid_retained_payload(
                "CCM residual-analysis extrema are reversed",
            ));
        }
        for (value, field) in [
            (
                &artifact.maximum_sampled_residual_u,
                "maximum residual abscissa",
            ),
            (
                &artifact.minimum_sampled_residual_u,
                "minimum residual abscissa",
            ),
        ] {
            if parse_retained_float(value, precision_bits, field)? <= 0u32 {
                return Err(invalid_retained_payload(format!(
                    "retained {field} must be positive"
                )));
            }
        }
        for bracket in &artifact.crossing_brackets {
            let left_u =
                parse_retained_float(&bracket.left_u, precision_bits, "crossing left abscissa")?;
            let right_u =
                parse_retained_float(&bracket.right_u, precision_bits, "crossing right abscissa")?;
            let left = parse_retained_float(
                &bracket.left_residual,
                precision_bits,
                "crossing left residual",
            )?;
            let right = parse_retained_float(
                &bracket.right_residual,
                precision_bits,
                "crossing right residual",
            )?;
            if left_u >= right_u || left == 0u32 || right == 0u32 || (left > 0u32) == (right > 0u32)
            {
                return Err(invalid_retained_payload(
                    "CCM residual crossing is not a strict sign-change bracket",
                ));
            }
        }
        for (entry, rule) in artifact.measurements.iter().zip(rules) {
            if entry.rule_family != rule.family()
                || entry.quadrature_rule != rule.rule()
                || entry.grid_variable != rule.variable().as_str()
                || entry.resolution != rule.resolution()
            {
                return Err(invalid_retained_payload(
                    "CCM residual-analysis rule does not match its request",
                ));
            }
            let absolute = parse_retained_float(
                &entry.absolute_residual_mass,
                precision_bits,
                "absolute residual mass",
            )?;
            let signed = parse_retained_float(
                &entry.signed_residual_mass,
                precision_bits,
                "signed residual mass",
            )?;
            let positive = parse_retained_float(
                &entry.positive_residual_mass,
                precision_bits,
                "positive residual mass",
            )?;
            let negative = parse_retained_float(
                &entry.negative_residual_mass,
                precision_bits,
                "negative residual mass",
            )?;
            if absolute < 0u32 || positive < 0u32 || negative < 0u32 {
                return Err(invalid_retained_payload(
                    "CCM residual masses must be nonnegative except for the signed mass",
                ));
            }
            let working = precision_bits.saturating_add(GUARD_BITS);
            let mut expected_positive = Float::with_val(working, &absolute + &signed);
            expected_positive /= 2u32;
            let mut expected_negative = Float::with_val(working, &absolute - &signed);
            expected_negative /= 2u32;
            if entry.positive_residual_mass
                != decimal(
                    &Float::with_val(precision_bits, expected_positive),
                    precision_bits,
                )
                || entry.negative_residual_mass
                    != decimal(
                        &Float::with_val(precision_bits, expected_negative),
                        precision_bits,
                    )
            {
                return Err(invalid_retained_payload(
                    "CCM one-sided residual masses do not reconstruct from absolute and signed mass",
                ));
            }
        }
        Ok(())
    }

    struct DecodedRetainedDistanceSource {
        eigenfunction: WeilEigenfunction,
        distances: Vec<WeightedGridValueHp>,
        u_values: Vec<Float>,
        f_values: Vec<Float>,
    }

    fn decode_retained_distance_source(
        profile: &PortableEigenfunctionProfile,
        distance: &PortableTargetDistance,
        lambda: &Float,
        alpha: &Float,
        rules: &[WeightedIntegrationRule],
        precision_bits: u32,
    ) -> Result<DecodedRetainedDistanceSource> {
        let parse = |text: &str, field: &str| -> Result<Float> {
            let parsed = Float::parse(text)
                .map_err(|error| anyhow::anyhow!("invalid retained {field}: {error}"))?;
            let value = Float::with_val(precision_bits, parsed);
            if !value.is_finite() {
                anyhow::bail!("retained {field} must be finite");
            }
            Ok(value)
        };
        let coefficients = profile
            .normalized_coefficients
            .iter()
            .map(|value| parse(value, "normalized coefficient"))
            .collect::<Result<Vec<_>>>()?;
        let eigenfunction =
            WeilEigenfunction::from_normalized_coefficients(&coefficients, lambda, precision_bits)?;
        let u_values = profile
            .u_values
            .iter()
            .map(|value| parse(value, "profile abscissa"))
            .collect::<Result<Vec<_>>>()?;
        let f_values = profile
            .f_values
            .iter()
            .map(|value| parse(value, "profile value"))
            .collect::<Result<Vec<_>>>()?;
        let distances = rules
            .iter()
            .zip(&distance.measurements)
            .map(|(rule, measurement)| {
                Ok(WeightedGridValueHp {
                    value: parse(&measurement.distance_to_target, "target distance")?,
                    lambda: Float::with_val(precision_bits, lambda),
                    alpha: Float::with_val(precision_bits, alpha),
                    rule: *rule,
                    precision_bits,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(DecodedRetainedDistanceSource {
            eigenfunction,
            distances,
            u_values,
            f_values,
        })
    }

    /// Retain the eigenfunction profile and the target distance for one CCM
    /// configuration as `ccm-distance` artifacts.
    ///
    /// This is the explicit, opt-in capture path. It performs the same
    /// measurement as [`ccm_distance_to_target_hp`] and additionally writes
    /// both artifacts through the supplied cache context, so a later run — or
    /// a different consumer entirely — reuses them instead of recomputing.
    ///
    /// `profile_steps` controls only how densely the retained profile is
    /// sampled; the distance itself always uses `steps`.
    pub fn capture_ccm_distance_via_cache(
        params: &crate::ccm::CcmParams,
        cfg: &crate::ccm::hp::HighPrecConfig,
        alpha: &Float,
        rules: &[WeightedIntegrationRule],
        profile_steps: usize,
        cache: &xc_cache::ArtifactCacheContext<'_>,
    ) -> Result<CcmTargetDistanceHp> {
        capture_ccm_distance_via_cache_internal(
            params,
            cfg,
            alpha,
            rules,
            profile_steps,
            cache,
            DerivedDistanceCapture::NONE,
        )
    }

    /// Retain the ordinary profile and target-distance artifacts plus a
    /// first-class `ccm_distance_resolution_evidence` artifact. The evidence
    /// records coefficient-tail diagnostics and same-rule Q/2Q refinement,
    /// continuing deterministically to 4Q only when the declared tolerance
    /// is not met.
    pub fn capture_ccm_distance_with_resolution_evidence_via_cache(
        params: &crate::ccm::CcmParams,
        cfg: &crate::ccm::hp::HighPrecConfig,
        alpha: &Float,
        rules: &[WeightedIntegrationRule],
        profile_steps: usize,
        cache: &xc_cache::ArtifactCacheContext<'_>,
    ) -> Result<CcmTargetDistanceHp> {
        capture_ccm_distance_via_cache_internal(
            params,
            cfg,
            alpha,
            rules,
            profile_steps,
            cache,
            DerivedDistanceCapture::RESOLUTION_ONLY,
        )
    }

    /// Retain profile and distance artifacts plus first-class target-residual
    /// diagnostics, without requesting resolution evidence.
    pub fn capture_ccm_distance_with_residual_analysis_via_cache(
        params: &crate::ccm::CcmParams,
        cfg: &crate::ccm::hp::HighPrecConfig,
        alpha: &Float,
        rules: &[WeightedIntegrationRule],
        profile_steps: usize,
        cache: &xc_cache::ArtifactCacheContext<'_>,
    ) -> Result<CcmTargetDistanceHp> {
        capture_ccm_distance_via_cache_internal(
            params,
            cfg,
            alpha,
            rules,
            profile_steps,
            cache,
            DerivedDistanceCapture::RESIDUAL_ONLY,
        )
    }

    /// Retain profile and distance artifacts plus a first-class
    /// `ccm_deviation_decomposition`, without resolution or residual evidence.
    ///
    /// The decomposition reads only the retained profile, so this also
    /// backfills onto configurations captured before the artifact existed.
    pub fn capture_ccm_distance_with_deviation_decomposition_via_cache(
        params: &crate::ccm::CcmParams,
        cfg: &crate::ccm::hp::HighPrecConfig,
        alpha: &Float,
        rules: &[WeightedIntegrationRule],
        profile_steps: usize,
        cache: &xc_cache::ArtifactCacheContext<'_>,
    ) -> Result<CcmTargetDistanceHp> {
        capture_ccm_distance_via_cache_internal(
            params,
            cfg,
            alpha,
            rules,
            profile_steps,
            cache,
            DerivedDistanceCapture::DECOMPOSITION_ONLY,
        )
    }

    /// Retain the established distance artifacts plus any combination of the
    /// derived kinds.
    ///
    /// The named wrappers are convenience presets over this entry point; the
    /// three flags compose freely, so all eight combinations are expressible
    /// and none silently drops a requested artifact.
    #[allow(clippy::too_many_arguments)]
    pub fn capture_ccm_distance_with_derived_via_cache(
        params: &crate::ccm::CcmParams,
        cfg: &crate::ccm::hp::HighPrecConfig,
        alpha: &Float,
        rules: &[WeightedIntegrationRule],
        profile_steps: usize,
        cache: &xc_cache::ArtifactCacheContext<'_>,
        resolution_evidence: bool,
        residual_analysis: bool,
        deviation_decomposition: bool,
    ) -> Result<CcmTargetDistanceHp> {
        capture_ccm_distance_via_cache_internal(
            params,
            cfg,
            alpha,
            rules,
            profile_steps,
            cache,
            DerivedDistanceCapture {
                resolution_evidence,
                residual_analysis,
                deviation_decomposition,
            },
        )
    }

    /// Maximum numerical distance capture plus the opt-in deviation
    /// decomposition.
    pub fn capture_ccm_distance_with_numerical_analysis_and_decomposition_via_cache(
        params: &crate::ccm::CcmParams,
        cfg: &crate::ccm::hp::HighPrecConfig,
        alpha: &Float,
        rules: &[WeightedIntegrationRule],
        profile_steps: usize,
        cache: &xc_cache::ArtifactCacheContext<'_>,
    ) -> Result<CcmTargetDistanceHp> {
        capture_ccm_distance_via_cache_internal(
            params,
            cfg,
            alpha,
            rules,
            profile_steps,
            cache,
            DerivedDistanceCapture::MAXIMUM_WITH_DECOMPOSITION,
        )
    }

    /// Maximum numerical distance capture: retain both resolution evidence and
    /// target-residual diagnostics alongside the established artifacts.
    pub fn capture_ccm_distance_with_numerical_analysis_via_cache(
        params: &crate::ccm::CcmParams,
        cfg: &crate::ccm::hp::HighPrecConfig,
        alpha: &Float,
        rules: &[WeightedIntegrationRule],
        profile_steps: usize,
        cache: &xc_cache::ArtifactCacheContext<'_>,
    ) -> Result<CcmTargetDistanceHp> {
        capture_ccm_distance_via_cache_internal(
            params,
            cfg,
            alpha,
            rules,
            profile_steps,
            cache,
            DerivedDistanceCapture::MAXIMUM,
        )
    }

    #[derive(Clone, Copy)]
    pub(crate) struct DerivedDistanceCapture {
        pub(crate) resolution_evidence: bool,
        pub(crate) residual_analysis: bool,
        pub(crate) deviation_decomposition: bool,
    }

    impl DerivedDistanceCapture {
        pub(crate) const NONE: Self = Self {
            resolution_evidence: false,
            residual_analysis: false,
            deviation_decomposition: false,
        };
        pub(crate) const RESOLUTION_ONLY: Self = Self {
            resolution_evidence: true,
            residual_analysis: false,
            deviation_decomposition: false,
        };
        pub(crate) const RESIDUAL_ONLY: Self = Self {
            resolution_evidence: false,
            residual_analysis: true,
            deviation_decomposition: false,
        };
        pub(crate) const DECOMPOSITION_ONLY: Self = Self {
            resolution_evidence: false,
            residual_analysis: false,
            deviation_decomposition: true,
        };
        /// Deliberately excludes the deviation decomposition. Adding a new
        /// artifact to a named capture level would make a `require_reuse`
        /// reproduction of an existing shard fail on a missing artifact, so
        /// the decomposition is opt-in like the response families.
        pub(crate) const MAXIMUM: Self = Self {
            resolution_evidence: true,
            residual_analysis: true,
            deviation_decomposition: false,
        };
        pub(crate) const MAXIMUM_WITH_DECOMPOSITION: Self = Self {
            resolution_evidence: true,
            residual_analysis: true,
            deviation_decomposition: true,
        };
    }

    fn capture_ccm_distance_via_cache_internal(
        params: &crate::ccm::CcmParams,
        cfg: &crate::ccm::hp::HighPrecConfig,
        alpha: &Float,
        rules: &[WeightedIntegrationRule],
        profile_steps: usize,
        cache: &xc_cache::ArtifactCacheContext<'_>,
        derived_capture: DerivedDistanceCapture,
    ) -> Result<CcmTargetDistanceHp> {
        use std::collections::BTreeMap;
        use xc_cache::{
            resolve_or_compute_json_artifact_with_dependencies, ArtifactExecutionCacheRequest,
            CacheError, CacheQuality, DependencyRef, SemanticKeyEnvelope, ToolkitVersion,
        };

        if rules.is_empty() {
            anyhow::bail!("distance capture requires at least one integration rule");
        }
        // Reject invalid resolutions before any eigenstate or quadrature work:
        // a zero step count would otherwise divide by zero deep inside HP
        // sampling, and the sampling loops convert counts to u32.
        for rule in rules {
            rule.validate()?;
            if u32::try_from(rule.resolution()).is_err() {
                anyhow::bail!(
                    "integration rule resolution {} exceeds the u32 sampling range",
                    rule.resolution()
                );
            }
            if derived_capture.resolution_evidence
                && matches!(rule, WeightedIntegrationRule::UniformGrid { .. })
            {
                let four_times = rule
                    .resolution()
                    .checked_mul(RESOLUTION_EVIDENCE_MAXIMUM_MULTIPLIER)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "4Q resolution overflows usize for Q = {}",
                            rule.resolution()
                        )
                    })?;
                if u32::try_from(four_times).is_err() {
                    anyhow::bail!("4Q resolution {four_times} exceeds the u32 sampling range");
                }
            }
        }
        if derived_capture.resolution_evidence
            && !rules
                .iter()
                .any(|rule| matches!(rule, WeightedIntegrationRule::UniformGrid { .. }))
        {
            anyhow::bail!("resolution evidence requires at least one uniform-grid rule");
        }
        if profile_steps == 0 {
            anyhow::bail!("distance capture requires a positive profile step count");
        }
        if u32::try_from(profile_steps).is_err() {
            anyhow::bail!("profile step count {profile_steps} exceeds the u32 sampling range");
        }
        // The ground state is resolved and the eigenfunction reconstructed
        // exactly once, then every rule is evaluated against it. Resolving per
        // rule would repeat manifest validation and eigenvector decoding for
        // each entry, which is the dominant cost at large N and high
        // precision even when every resolution is a cache hit.
        let prec = cfg.precision_bits;
        let working = prec.saturating_add(GUARD_BITS);
        // The opaque digest is safe to retain; the private specification and
        // its coefficients never enter a semantic envelope or payload.
        let target_spec = crate::target::TargetProfileSpec::from_environment()?;
        let target_definition_digest = target_spec.digest()?;
        let lambda_sq_identity = lambda_squared_identity(params);
        let lambda_sq = if params.lambda_sq.is_integer {
            Float::with_val(working, params.lambda_sq.value_u64)
        } else {
            Float::with_val(working, params.lambda_sq.value_f64)
        };
        if lambda_sq <= 1u32 {
            anyhow::bail!(
                "distance capture integrates over [1, λ] and needs λ² > 1 (got {})",
                params.lambda_squared()
            );
        }
        let lambda = lambda_sq.sqrt();
        let variable = rules[0].variable();
        let canonical_state =
            crate::ccm::hp::resolve_canonical_even_eigenstate_via_cache(params, cfg, cache)?;
        let eigenpair_content_digest = canonical_state.manifest.content_digest.0.clone();
        let eigenpair_dependency = || {
            vec![DependencyRef {
                key: canonical_state.manifest.key.clone(),
                content_digest: canonical_state.manifest.content_digest.clone(),
                required_quality: CacheQuality::Validated,
            }]
        };

        // ------------------------------------------------------------------
        // Numerical state, computed lazily. Under reuse modes the closures
        // below only run this when at least one artifact misses; under
        // refresh/verify/disabled modes it runs eagerly so the established
        // compute-and-compare semantics are preserved.
        //
        // Within one computation the eigenfunction is evaluated once per
        // abscissa: the distance pass records f(u) in evaluation order and
        // the norm pass replays the recorded values, so both accumulators
        // consume bit-identical function values while each retains its
        // established reduction order.
        // ------------------------------------------------------------------
        struct ComputedCapture {
            eigenfunction: WeilEigenfunction,
            eigenvalue: Float,
            distances: Vec<WeightedGridValueHp>,
            norm_values: Vec<Float>,
            signed_residual_values: Option<Vec<Float>>,
            base_value_samples: Vec<Vec<(Float, Float)>>,
            u_values: Vec<Float>,
            f_values: Vec<Float>,
        }

        let compute_state = || -> Result<ComputedCapture> {
            let expected = params.matrix_size();
            if canonical_state.eigenvector.len() != expected {
                anyhow::bail!(
                    "canonical CCM even eigenvector has dimension {}, expected 2N+1 = {expected}",
                    canonical_state.eigenvector.len(),
                );
            }
            let eigenfunction = WeilEigenfunction::from_v_basis(
                &canonical_state.eigenvector,
                params.n_modes,
                &lambda,
                prec,
            )?;

            // One Gauss--Legendre table per unique (points, working) request,
            // shared by every distance and norm measurement below.
            let mut gl_tables = SharedGlTables::new();
            gl_tables.preload_managed(rules, prec, cache)?;
            let mut distances = Vec::with_capacity(rules.len());
            let mut norm_values = Vec::with_capacity(rules.len());
            let mut base_value_samples = Vec::with_capacity(rules.len());
            let mut signed_residual_values = derived_capture
                .residual_analysis
                .then(|| Vec::with_capacity(rules.len()));
            for rule in rules {
                let recorded_values = std::cell::RefCell::new(Vec::new());
                let recorded_samples = std::cell::RefCell::new(Vec::new());
                let retain_base_samples = derived_capture.resolution_evidence
                    && matches!(
                        *rule,
                        WeightedIntegrationRule::UniformGrid {
                            scheme: UniformGridScheme::LeftRiemann
                                | UniformGridScheme::RightRiemann
                                | UniformGridScheme::Trapezoid,
                            ..
                        }
                    );
                let distance = distance_to_target_with_tables_bound(
                    |u: &Float| {
                        let value = eigenfunction.eval(u);
                        recorded_values.borrow_mut().push(value.clone());
                        if retain_base_samples {
                            recorded_samples
                                .borrow_mut()
                                .push((u.clone(), value.clone()));
                        }
                        value
                    },
                    &lambda,
                    alpha,
                    *rule,
                    prec,
                    Some(&mut gl_tables),
                    Some(&target_definition_digest),
                )?;
                let recorded_values = recorded_values.into_inner();
                if let Some(signed_values) = &mut signed_residual_values {
                    let signed_cursor = std::cell::Cell::new(0usize);
                    let signed_replay_overrun = std::cell::Cell::new(false);
                    let signed = signed_residual_to_target_with_tables(
                        |_u: &Float| {
                            let index = signed_cursor.get();
                            signed_cursor.set(index + 1);
                            match recorded_values.get(index) {
                                Some(value) => value.clone(),
                                None => {
                                    signed_replay_overrun.set(true);
                                    Float::with_val(working, 0u32)
                                }
                            }
                        },
                        &lambda,
                        alpha,
                        *rule,
                        prec,
                        Some(&mut gl_tables),
                        Some(&target_definition_digest),
                    )?;
                    if signed_replay_overrun.get() || signed_cursor.get() != recorded_values.len() {
                        anyhow::bail!(
                            "fused distance/signed-residual replay expected {} evaluations, consumed {}",
                            recorded_values.len(),
                            signed_cursor.get()
                        );
                    }
                    signed_values.push(signed);
                }
                let cursor = std::cell::Cell::new(0usize);
                let replay_overrun = std::cell::Cell::new(false);
                let norm = weighted_alpha_norm_with_tables(
                    |_u: &Float| {
                        let index = cursor.get();
                        cursor.set(index + 1);
                        match recorded_values.get(index) {
                            Some(value) => value.clone(),
                            None => {
                                replay_overrun.set(true);
                                Float::with_val(working, 0u32)
                            }
                        }
                    },
                    &lambda,
                    alpha,
                    *rule,
                    prec,
                    Some(&mut gl_tables),
                )?;
                if replay_overrun.get() || cursor.get() != recorded_values.len() {
                    anyhow::bail!(
                        "fused distance/norm replay expected {} evaluations, consumed {}",
                        recorded_values.len(),
                        cursor.get()
                    );
                }
                distances.push(distance);
                norm_values.push(norm.value);
                base_value_samples.push(recorded_samples.into_inner());
            }

            // The profile grid follows the first rule's variable; the
            // retained coefficients make any other choice recomputable
            // downstream.
            let (u_values, f_values) =
                sample_profile(&eigenfunction, &lambda, profile_steps, variable, prec);
            Ok(ComputedCapture {
                eigenvalue: Float::with_val(prec, &canonical_state.eigenvalue),
                eigenfunction,
                distances,
                norm_values,
                signed_residual_values,
                base_value_samples,
                u_values,
                f_values,
            })
        };

        let build_profile_payload = |state: &ComputedCapture| PortableEigenfunctionProfile {
            schema_version: 1,
            lambda_squared: lambda_sq_identity.clone(),
            n_modes: params.n_modes,
            precision_bits: prec,
            grid_variable: variable.as_str().to_owned(),
            sample_count: state.u_values.len(),
            normalization: "f(1)=1".to_owned(),
            u_values: state.u_values.iter().map(|v| decimal(v, prec)).collect(),
            f_values: state.f_values.iter().map(|v| decimal(v, prec)).collect(),
            normalized_coefficients: state
                .eigenfunction
                .normalized_coefficients()
                .iter()
                .map(|v| decimal(v, prec))
                .collect(),
        };
        let build_distance_payload = |state: &ComputedCapture| PortableTargetDistance {
            schema_version: 2,
            target_definition_digest: target_definition_digest.clone(),
            lambda_squared: lambda_sq_identity.clone(),
            n_modes: params.n_modes,
            precision_bits: prec,
            alpha: decimal(alpha, prec),
            measurements: rules
                .iter()
                .zip(state.distances.iter().zip(&state.norm_values))
                .map(|(rule, (distance, norm))| PortableRuleMeasurement {
                    rule_family: rule.family().to_owned(),
                    quadrature_rule: rule.rule().to_owned(),
                    grid_variable: rule.variable().as_str().to_owned(),
                    resolution: rule.resolution(),
                    distance_to_target: decimal(&distance.value, prec),
                    eigenfunction_norm: decimal(norm, prec),
                })
                .collect(),
            eigenvalue: decimal(&state.eigenvalue, prec),
        };
        let build_resolution_evidence_payload = |state: &ComputedCapture| {
            compute_distance_resolution_evidence_with_samples(
                &state.eigenfunction,
                &lambda,
                alpha,
                rules,
                ResolutionEvidenceEvaluationSource {
                    base_distances: &state.distances,
                    base_value_samples: Some(&state.base_value_samples),
                },
                &target_definition_digest,
                &lambda_sq_identity,
                prec,
            )
        };
        let build_residual_analysis_payload = |state: &ComputedCapture| {
            compute_target_residual_analysis(TargetResidualAnalysisSource {
                eigenfunction: &state.eigenfunction,
                lambda: &lambda,
                alpha,
                rules,
                base_distances: &state.distances,
                u_values: &state.u_values,
                f_values: &state.f_values,
                precomputed_signed_residuals: state.signed_residual_values.as_deref(),
                target_definition_digest: &target_definition_digest,
                lambda_squared: &lambda_sq_identity,
                sampling_variable: variable,
                precision_bits: prec,
            })
        };

        let profile_key = SemanticKeyEnvelope {
            schema_version: 1,
            artifact_kind: "ccm_eigenfunction_profile".to_owned(),
            mathematical_semantics_version: "ccm-eigenfunction-profile-v0.14.1-v2".to_owned(),
            resolved_mathematical_parameters: serde_json::json!({
                "lambda_squared": lambda_sq_identity,
                "n_modes": params.n_modes,
                "precision_bits": prec,
                "grid_variable": variable.as_str(),
                "profile_steps": profile_steps,
                "eigenpair_content_digest": eigenpair_content_digest,
                "definition": "even CCM ground eigenfunction sampled on [1, lambda]"
            }),
            normalization: Some("f(1)=1".to_owned()),
            target: Some("finite_ccm_even_ground_eigenfunction".to_owned()),
            subspace: Some("even".to_owned()),
            source_data_identities: BTreeMap::new(),
            algorithm_semantics: Some(
                "even_cosine_reconstruction_from_canonical_weil_eigenpair_v2".to_owned(),
            ),
        };
        let profile_logical_key = format!(
            "ccm/eigenfunction-profile/{}/{}/{}/{}/{}",
            lambda_sq_identity,
            params.n_modes,
            prec,
            variable.as_str(),
            profile_steps
        );
        let profile_request = ArtifactExecutionCacheRequest {
            operation: "ccm.eigenfunction_profile.resolve_or_compute",
            semantic_key: &profile_key,
            logical_key: &profile_logical_key,
            resolver: cache.resolver,
            reference_resolver: cache.reference_resolver,
            acceptance: cache.acceptance,
            ordered_overlays: cache.ordered_overlays.clone(),
            mode: cache.mode,
            write_on_miss: cache.write_on_miss,
            write_visibility: cache.write_visibility,
            produced_quality: CacheQuality::Validated,
            producer_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION"))?,
            minimum_reader_version: ToolkitVersion::parse("0.14.1")?,
            maximum_reader_version: None,
            tags: BTreeMap::from([
                ("domain".to_owned(), "ccm".to_owned()),
                ("artifact".to_owned(), "eigenfunction_profile".to_owned()),
            ]),
            provenance_digest: None,
            production_sink: cache.production_sink,
        };

        let distance_key = SemanticKeyEnvelope {
            schema_version: 1,
            artifact_kind: "ccm_target_distance".to_owned(),
            mathematical_semantics_version: "ccm-runtime-target-distance-v0.14.1-v3".to_owned(),
            resolved_mathematical_parameters: serde_json::json!({
                "target_definition_digest": target_definition_digest,
                "lambda_squared": lambda_sq_identity,
                "n_modes": params.n_modes,
                "precision_bits": prec,
                "eigenpair_content_digest": eigenpair_content_digest,
                "alpha": decimal(alpha, prec),
                "rules": rules
                    .iter()
                    .map(|rule| {
                        serde_json::json!({
                            "rule_family": rule.family(),
                            "quadrature_rule": rule.rule(),
                            "grid_variable": rule.variable().as_str(),
                            "resolution": rule.resolution(),
                        })
                    })
                    .collect::<Vec<_>>(),
                "definition": "weighted absolute distance to runtime-supplied normalized target"
            }),
            normalization: Some("f(1)=1".to_owned()),
            target: Some("ccm_target_distance".to_owned()),
            subspace: Some("even".to_owned()),
            source_data_identities: BTreeMap::new(),
            algorithm_semantics: Some(
                "canonical_eigenpair_weighted_absolute_difference_v2".to_owned(),
            ),
        };
        let rule_signature = rules
            .iter()
            .map(|rule| {
                format!(
                    "{}-{}-{}",
                    rule.rule(),
                    rule.variable().as_str(),
                    rule.resolution()
                )
            })
            .collect::<Vec<_>>()
            .join("_");
        let distance_logical_key = format!(
            "ccm/target-distance/{}/{}/{}/{}",
            lambda_sq_identity, params.n_modes, prec, rule_signature
        );
        let distance_request = ArtifactExecutionCacheRequest {
            operation: "ccm.target_distance.resolve_or_compute",
            semantic_key: &distance_key,
            logical_key: &distance_logical_key,
            resolver: cache.resolver,
            reference_resolver: cache.reference_resolver,
            acceptance: cache.acceptance,
            ordered_overlays: cache.ordered_overlays.clone(),
            mode: cache.mode,
            write_on_miss: cache.write_on_miss,
            write_visibility: cache.write_visibility,
            produced_quality: CacheQuality::Validated,
            producer_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION"))?,
            minimum_reader_version: ToolkitVersion::parse("0.14.1")?,
            maximum_reader_version: None,
            tags: BTreeMap::from([
                ("domain".to_owned(), "ccm".to_owned()),
                ("artifact".to_owned(), "target_distance".to_owned()),
            ]),
            provenance_digest: None,
            production_sink: cache.production_sink,
        };

        let uniform_rule_parameters = rules
            .iter()
            .filter(|rule| matches!(rule, WeightedIntegrationRule::UniformGrid { .. }))
            .map(|rule| {
                serde_json::json!({
                    "rule_family": rule.family(),
                    "quadrature_rule": rule.rule(),
                    "grid_variable": rule.variable().as_str(),
                    "base_resolution": rule.resolution(),
                })
            })
            .collect::<Vec<_>>();
        let evidence_key = SemanticKeyEnvelope {
            schema_version: 1,
            artifact_kind: "ccm_distance_resolution_evidence".to_owned(),
            mathematical_semantics_version: "ccm-runtime-target-resolution-evidence-v0.14.1-v3"
                .to_owned(),
            resolved_mathematical_parameters: serde_json::json!({
                "target_definition_digest": target_definition_digest,
                "lambda_squared": lambda_sq_identity,
                "n_modes": params.n_modes,
                "precision_bits": prec,
                "eigenpair_content_digest": eigenpair_content_digest,
                "alpha": decimal(alpha, prec),
                "uniform_grid_rules": uniform_rule_parameters,
                "coefficient_thresholds": RESOLUTION_EVIDENCE_THRESHOLD_DECADES
                    .iter()
                    .map(|decades| format!("1e-{decades}"))
                    .collect::<Vec<_>>(),
                "refinement_factor": RESOLUTION_EVIDENCE_REFINEMENT_FACTOR,
                "maximum_refinement_multiplier": RESOLUTION_EVIDENCE_MAXIMUM_MULTIPLIER,
                "relative_tolerance": RESOLUTION_EVIDENCE_RELATIVE_TOLERANCE,
                "relative_difference_denominator": "absolute_finer_distance",
                "zero_denominator_fallback": "absolute_difference",
            }),
            normalization: Some("f(1)=1".to_owned()),
            target: Some("ccm_target_distance_resolution".to_owned()),
            subspace: Some("even".to_owned()),
            source_data_identities: BTreeMap::new(),
            algorithm_semantics: Some(
                "canonical_eigenpair_coefficient_tail_and_same_uniform_rule_q_2q_conditional_4q_v2"
                    .to_owned(),
            ),
        };
        let uniform_rule_signature = rules
            .iter()
            .filter(|rule| matches!(rule, WeightedIntegrationRule::UniformGrid { .. }))
            .map(|rule| {
                format!(
                    "{}-{}-{}",
                    rule.rule(),
                    rule.variable().as_str(),
                    rule.resolution()
                )
            })
            .collect::<Vec<_>>()
            .join("_");
        let evidence_logical_key = format!(
            "ccm/distance-resolution-evidence/{}/{}/{}/{}",
            lambda_sq_identity, params.n_modes, prec, uniform_rule_signature
        );
        let evidence_request = ArtifactExecutionCacheRequest {
            operation: "ccm.distance_resolution_evidence.resolve_or_compute",
            semantic_key: &evidence_key,
            logical_key: &evidence_logical_key,
            resolver: cache.resolver,
            reference_resolver: cache.reference_resolver,
            acceptance: cache.acceptance,
            ordered_overlays: cache.ordered_overlays.clone(),
            mode: cache.mode,
            write_on_miss: cache.write_on_miss,
            write_visibility: cache.write_visibility,
            produced_quality: CacheQuality::Validated,
            producer_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION"))?,
            minimum_reader_version: ToolkitVersion::parse("0.14.1")?,
            maximum_reader_version: None,
            tags: BTreeMap::from([
                ("domain".to_owned(), "ccm".to_owned()),
                (
                    "artifact".to_owned(),
                    "distance_resolution_evidence".to_owned(),
                ),
            ]),
            provenance_digest: None,
            production_sink: cache.production_sink,
        };

        let residual_key = SemanticKeyEnvelope {
            schema_version: 1,
            artifact_kind: "ccm_target_residual_analysis".to_owned(),
            mathematical_semantics_version: "ccm-runtime-target-residual-analysis-v0.14.1-v3"
                .to_owned(),
            resolved_mathematical_parameters: serde_json::json!({
                "target_definition_digest": target_definition_digest,
                "lambda_squared": lambda_sq_identity,
                "n_modes": params.n_modes,
                "precision_bits": prec,
                "eigenpair_content_digest": eigenpair_content_digest,
                "alpha": decimal(alpha, prec),
                "rules": rules
                    .iter()
                    .map(|rule| serde_json::json!({
                        "rule_family": rule.family(),
                        "quadrature_rule": rule.rule(),
                        "grid_variable": rule.variable().as_str(),
                        "resolution": rule.resolution(),
                    }))
                    .collect::<Vec<_>>(),
                "sampling_grid_variable": variable.as_str(),
                "profile_steps": profile_steps,
                "crossing_bracket_policy": "adjacent_nonzero_profile_samples_v1",
                "one_sided_mass_derivation":
                    "positive=(absolute+signed)/2;negative=(absolute-signed)/2",
                "mass_consistency_policy": RESIDUAL_MASS_CONSISTENCY_POLICY,
                "piecewise_integration": false,
            }),
            normalization: Some("f(1)=1".to_owned()),
            target: Some("ccm_target_residual_analysis".to_owned()),
            subspace: Some("even".to_owned()),
            source_data_identities: BTreeMap::new(),
            algorithm_semantics: Some(
                "canonical_eigenpair_profile_sampled_sign_structure_and_same_rule_signed_mass_v2"
                    .to_owned(),
            ),
        };
        let residual_logical_key = format!(
            "ccm/target-residual-analysis/{}/{}/{}/{}/{}",
            lambda_sq_identity, params.n_modes, prec, profile_steps, rule_signature
        );
        let residual_request = ArtifactExecutionCacheRequest {
            operation: "ccm.target_residual_analysis.resolve_or_compute",
            semantic_key: &residual_key,
            logical_key: &residual_logical_key,
            resolver: cache.resolver,
            reference_resolver: cache.reference_resolver,
            acceptance: cache.acceptance,
            ordered_overlays: cache.ordered_overlays.clone(),
            mode: cache.mode,
            write_on_miss: cache.write_on_miss,
            write_visibility: cache.write_visibility,
            produced_quality: CacheQuality::Validated,
            producer_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION"))?,
            minimum_reader_version: ToolkitVersion::parse("0.14.1")?,
            maximum_reader_version: None,
            tags: BTreeMap::from([
                ("domain".to_owned(), "ccm".to_owned()),
                ("artifact".to_owned(), "target_residual_analysis".to_owned()),
            ]),
            provenance_digest: None,
            production_sink: cache.production_sink,
        };

        let decomposition_key = SemanticKeyEnvelope {
            schema_version: 1,
            artifact_kind: "ccm_deviation_decomposition".to_owned(),
            mathematical_semantics_version: "ccm-runtime-target-decomposition-v0.14.1-v4"
                .to_owned(),
            resolved_mathematical_parameters: serde_json::json!({
                "target_definition_digest": target_definition_digest,
                "lambda_squared": lambda_sq_identity,
                "n_modes": params.n_modes,
                "precision_bits": prec,
                "eigenpair_content_digest": eigenpair_content_digest,
                "sampling_grid_variable": variable.as_str(),
                "profile_steps": profile_steps,
                "auxiliary_parameter_condition":
                    "runtime specification fixes the auxiliary endpoint",
                "quadrature_rule": DEVIATION_QUADRATURE_RULE,
                "sign_convention": DEVIATION_SIGN_CONVENTION,
                "metrics": [
                    crate::deviation::DeviationMetric::FactorWeighted.as_str(),
                    crate::deviation::DeviationMetric::IntegrandWeighted.as_str(),
                ],
                "crossing_policy": "vanishing_amplitude_is_recorded_not_rejected",
            }),
            normalization: Some("f(1)=1".to_owned()),
            target: Some("ccm_deviation_decomposition".to_owned()),
            subspace: Some("even".to_owned()),
            source_data_identities: BTreeMap::new(),
            algorithm_semantics: Some(
                "canonical_eigenpair_profile_grid_trapezoid_projection_onto_runtime_auxiliary_profile_v2".to_owned(),
            ),
        };
        let decomposition_logical_key = format!(
            "ccm/deviation-decomposition/{}/{}/{}/{}",
            lambda_sq_identity, params.n_modes, prec, profile_steps
        );
        let decomposition_request = ArtifactExecutionCacheRequest {
            operation: "ccm.deviation_decomposition.resolve_or_compute",
            semantic_key: &decomposition_key,
            logical_key: &decomposition_logical_key,
            resolver: cache.resolver,
            reference_resolver: cache.reference_resolver,
            acceptance: cache.acceptance,
            ordered_overlays: cache.ordered_overlays.clone(),
            mode: cache.mode,
            write_on_miss: cache.write_on_miss,
            write_visibility: cache.write_visibility,
            produced_quality: CacheQuality::Validated,
            producer_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION"))?,
            minimum_reader_version: ToolkitVersion::parse("0.14.1")?,
            maximum_reader_version: None,
            tags: BTreeMap::from([
                ("domain".to_owned(), "ccm".to_owned()),
                ("artifact".to_owned(), "deviation_decomposition".to_owned()),
            ]),
            provenance_digest: None,
            production_sink: cache.production_sink,
        };

        let structural_profile_check = |artifact: &PortableEigenfunctionProfile| {
            validate_portable_eigenfunction_profile(
                artifact,
                &lambda_sq_identity,
                params.n_modes,
                prec,
                variable,
                profile_steps,
            )
        };
        let structural_distance_check = |artifact: &PortableTargetDistance| {
            validate_portable_target_distance(
                artifact,
                TargetDistanceValidationRequest {
                    target_definition_digest: &target_definition_digest,
                    lambda_squared: &lambda_sq_identity,
                    n_modes: params.n_modes,
                    precision_bits: prec,
                    alpha,
                    rules,
                    expected_eigenvalue: &canonical_state.eigenvalue,
                },
            )
        };
        let structural_resolution_evidence_check =
            |artifact: &PortableDistanceResolutionEvidence| {
                validate_portable_distance_resolution_evidence(
                    artifact,
                    &target_definition_digest,
                    &lambda_sq_identity,
                    params.n_modes,
                    prec,
                    alpha,
                    rules,
                )
            };
        let structural_residual_analysis_check = |artifact: &PortableTargetResidualAnalysis| {
            validate_portable_target_residual_analysis(
                artifact,
                TargetResidualAnalysisValidationRequest {
                    target_definition_digest: &target_definition_digest,
                    lambda_squared: &lambda_sq_identity,
                    n_modes: params.n_modes,
                    precision_bits: prec,
                    alpha,
                    rules,
                    variable,
                    profile_steps,
                },
            )
        };

        if cache.mode.consults_cache_for_result_reuse() {
            // Reuse modes: compute lazily, only when an artifact misses. A
            // full hit performs no sector resolution, no eigenfunction
            // reconstruction, no quadrature, and no sampling; the returned
            // measurement is decoded from the retained artifact at its
            // stored decimal precision.
            let state: std::cell::RefCell<Option<ComputedCapture>> = std::cell::RefCell::new(None);
            let ensure = |state: &std::cell::RefCell<Option<ComputedCapture>>|
             -> std::result::Result<(), CacheError> {
                if state.borrow().is_none() {
                    let computed = compute_state().map_err(|error| {
                        CacheError::InvalidTransition(format!(
                            "distance capture computation failed: {error}"
                        ))
                    })?;
                    *state.borrow_mut() = Some(computed);
                }
                Ok(())
            };

            let resolved_profile = resolve_or_compute_json_artifact_with_dependencies(
                &profile_request,
                || {
                    ensure(&state)?;
                    let guard = state.borrow();
                    Ok((
                        build_profile_payload(guard.as_ref().expect("state ensured")),
                        eigenpair_dependency(),
                    ))
                },
                structural_profile_check,
            )?;
            structural_profile_check(&resolved_profile.value)?;

            let resolved_distance = resolve_or_compute_json_artifact_with_dependencies(
                &distance_request,
                || {
                    ensure(&state)?;
                    let guard = state.borrow();
                    Ok((
                        build_distance_payload(guard.as_ref().expect("state ensured")),
                        eigenpair_dependency(),
                    ))
                },
                structural_distance_check,
            )?;
            structural_distance_check(&resolved_distance.value)?;

            if derived_capture.resolution_evidence {
                let resolved_evidence = resolve_or_compute_json_artifact_with_dependencies(
                    &evidence_request,
                    || {
                        let guard = state.borrow();
                        if let Some(computed) = guard.as_ref() {
                            return build_resolution_evidence_payload(computed)
                                .map(|payload| (payload, eigenpair_dependency()))
                                .map_err(|error| {
                                    CacheError::InvalidTransition(format!(
                                        "distance resolution-evidence computation failed: {error}"
                                    ))
                                });
                        }
                        drop(guard);
                        let retained = decode_retained_distance_source(
                            &resolved_profile.value,
                            &resolved_distance.value,
                            &lambda,
                            alpha,
                            rules,
                            prec,
                        )
                        .map_err(|error| {
                            CacheError::InvalidTransition(format!(
                                "retained distance source could not be reconstructed: {error}"
                            ))
                        })?;
                        compute_distance_resolution_evidence(
                            &retained.eigenfunction,
                            &lambda,
                            alpha,
                            rules,
                            &retained.distances,
                            &target_definition_digest,
                            &lambda_sq_identity,
                            prec,
                        )
                        .map(|payload| (payload, eigenpair_dependency()))
                        .map_err(|error| {
                            CacheError::InvalidTransition(format!(
                                "distance resolution-evidence backfill failed: {error}"
                            ))
                        })
                    },
                    structural_resolution_evidence_check,
                )?;
                structural_resolution_evidence_check(&resolved_evidence.value)?;
            }

            if derived_capture.deviation_decomposition {
                let expected_samples = resolved_profile.value.u_values.len();
                let check = |artifact: &PortableDeviationDecomposition| {
                    structural_deviation_decomposition_check(
                        artifact,
                        &lambda_sq_identity,
                        params.n_modes,
                        prec,
                        variable,
                        expected_samples,
                        &target_definition_digest,
                    )
                };
                let resolved_decomposition = resolve_or_compute_json_artifact_with_dependencies(
                    &decomposition_request,
                    || {
                        compute_deviation_decomposition_payload(&resolved_profile.value, prec)
                            .map(|payload| (payload, eigenpair_dependency()))
                            .map_err(|error| {
                                CacheError::InvalidTransition(format!(
                                    "deviation decomposition computation failed: {error}"
                                ))
                            })
                    },
                    check,
                )?;
                check(&resolved_decomposition.value)?;
            }

            if derived_capture.residual_analysis {
                let resolved_residual = resolve_or_compute_json_artifact_with_dependencies(
                    &residual_request,
                    || {
                        let guard = state.borrow();
                        if let Some(computed) = guard.as_ref() {
                            return build_residual_analysis_payload(computed)
                                .map(|payload| (payload, eigenpair_dependency()))
                                .map_err(|error| {
                                    CacheError::InvalidTransition(format!(
                                        "target residual-analysis computation failed: {error}"
                                    ))
                                });
                        }
                        drop(guard);
                        let retained = decode_retained_distance_source(
                            &resolved_profile.value,
                            &resolved_distance.value,
                            &lambda,
                            alpha,
                            rules,
                            prec,
                        )
                        .map_err(|error| {
                            CacheError::InvalidTransition(format!(
                                "retained distance source could not be reconstructed: {error}"
                            ))
                        })?;
                        let mut gl_tables = SharedGlTables::new();
                        gl_tables
                            .preload_managed(rules, prec, cache)
                            .map_err(|error| {
                                CacheError::InvalidTransition(format!(
                                    "managed distance quadrature resolution failed: {error}"
                                ))
                            })?;
                        compute_target_residual_analysis_with_tables(
                            TargetResidualAnalysisSource {
                                eigenfunction: &retained.eigenfunction,
                                lambda: &lambda,
                                alpha,
                                rules,
                                base_distances: &retained.distances,
                                u_values: &retained.u_values,
                                f_values: &retained.f_values,
                                precomputed_signed_residuals: None,
                                target_definition_digest: &target_definition_digest,
                                lambda_squared: &lambda_sq_identity,
                                sampling_variable: variable,
                                precision_bits: prec,
                            },
                            Some(&mut gl_tables),
                        )
                        .map(|payload| (payload, eigenpair_dependency()))
                        .map_err(|error| {
                            CacheError::InvalidTransition(format!(
                                "target residual-analysis backfill failed: {error}"
                            ))
                        })
                    },
                    structural_residual_analysis_check,
                )?;
                structural_residual_analysis_check(&resolved_residual.value)?;
            }

            if let Some(state) = state.into_inner() {
                return Ok(CcmTargetDistanceHp {
                    lambda_squared: params.lambda_squared(),
                    n_modes: params.n_modes,
                    eigenvalue: state.eigenvalue,
                    distances: state.distances,
                });
            }
            // Full hit: decode the measurement from the retained artifact.
            let parse = |text: &str| -> Result<Float> {
                let parsed = Float::parse(text)
                    .map_err(|error| anyhow::anyhow!("invalid retained decimal: {error}"))?;
                Ok(Float::with_val(prec, parsed))
            };
            let mut distances = Vec::with_capacity(rules.len());
            for (rule, entry) in rules.iter().zip(&resolved_distance.value.measurements) {
                distances.push(WeightedGridValueHp {
                    value: parse(&entry.distance_to_target)?,
                    lambda: Float::with_val(prec, &lambda),
                    alpha: Float::with_val(prec, alpha),
                    rule: *rule,
                    precision_bits: prec,
                });
            }
            return Ok(CcmTargetDistanceHp {
                lambda_squared: params.lambda_squared(),
                n_modes: params.n_modes,
                eigenvalue: parse(&resolved_distance.value.eigenvalue)?,
                distances,
            });
        }

        // Refresh / verify / disabled modes: compute eagerly and retain the
        // established compute-and-compare replay semantics.
        let state = compute_state()?;
        let profile_payload = build_profile_payload(&state);
        let resolved_profile = resolve_or_compute_json_artifact_with_dependencies(
            &profile_request,
            || Ok((profile_payload.clone(), eigenpair_dependency())),
            |artifact| {
                if artifact != &profile_payload {
                    return Err(CacheError::InvalidManifest(
                        "CCM eigenfunction profile does not replay from its eigenstate".to_owned(),
                    ));
                }
                Ok(())
            },
        )?;
        if resolved_profile.value != profile_payload {
            anyhow::bail!("resolved CCM eigenfunction profile disagrees with replayed values");
        }

        let distance_payload = build_distance_payload(&state);
        let resolved_distance = resolve_or_compute_json_artifact_with_dependencies(
            &distance_request,
            || Ok((distance_payload.clone(), eigenpair_dependency())),
            |artifact| {
                if artifact != &distance_payload {
                    return Err(CacheError::InvalidManifest(
                        "CCM target distance does not replay under its stated convention"
                            .to_owned(),
                    ));
                }
                Ok(())
            },
        )?;
        if resolved_distance.value != distance_payload {
            anyhow::bail!("resolved CCM target distance disagrees with replayed values");
        }

        if derived_capture.resolution_evidence {
            let evidence_payload = build_resolution_evidence_payload(&state)?;
            let resolved_evidence = resolve_or_compute_json_artifact_with_dependencies(
                &evidence_request,
                || Ok((evidence_payload.clone(), eigenpair_dependency())),
                |artifact| {
                    if artifact != &evidence_payload {
                        return Err(CacheError::InvalidManifest(
                            "CCM distance resolution evidence does not replay under its stated policy"
                                .to_owned(),
                        ));
                    }
                    Ok(())
                },
            )?;
            if resolved_evidence.value != evidence_payload {
                anyhow::bail!(
                    "resolved CCM distance resolution evidence disagrees with replayed values"
                );
            }
        }

        if derived_capture.deviation_decomposition {
            let expected_samples = resolved_profile.value.u_values.len();
            let check = |artifact: &PortableDeviationDecomposition| {
                structural_deviation_decomposition_check(
                    artifact,
                    &lambda_sq_identity,
                    params.n_modes,
                    prec,
                    variable,
                    expected_samples,
                    &target_definition_digest,
                )
            };
            let resolved_decomposition = resolve_or_compute_json_artifact_with_dependencies(
                &decomposition_request,
                || {
                    compute_deviation_decomposition_payload(&resolved_profile.value, prec)
                        .map_err(|error| {
                            CacheError::InvalidTransition(format!(
                                "deviation decomposition computation failed: {error}"
                            ))
                        })
                        .map(|payload| (payload, eigenpair_dependency()))
                },
                check,
            )?;
            check(&resolved_decomposition.value)?;
        }

        if derived_capture.residual_analysis {
            let residual_payload = build_residual_analysis_payload(&state)?;
            let resolved_residual = resolve_or_compute_json_artifact_with_dependencies(
                &residual_request,
                || Ok((residual_payload.clone(), eigenpair_dependency())),
                |artifact| {
                    if artifact != &residual_payload {
                        return Err(CacheError::InvalidManifest(
                            "CCM target residual analysis does not replay under its stated policy"
                                .to_owned(),
                        ));
                    }
                    Ok(())
                },
            )?;
            if resolved_residual.value != residual_payload {
                anyhow::bail!(
                    "resolved CCM target residual analysis disagrees with replayed values"
                );
            }
        }

        Ok(CcmTargetDistanceHp {
            lambda_squared: params.lambda_squared(),
            n_modes: params.n_modes,
            eigenvalue: state.eigenvalue,
            distances: state.distances,
        })
    }

    /// One `D_alpha` value under one integration rule.
    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct PortableRuleDistance {
        /// Rule family: `uniform_grid` or `gauss_legendre`.
        pub rule_family: String,
        /// Specific rule within the family.
        pub quadrature_rule: String,
        pub grid_variable: String,
        /// Grid cells or quadrature points, per the rule.
        pub resolution: usize,
        /// `D_α(N, M; λ)` under this rule.
        pub distance: String,
    }

    /// Portable inter-discretization distances for one `λ²`.
    ///
    /// `D_α` is symmetric in its two cutoffs, so the pair is stored in
    /// ascending order and the artifact identity is canonical: measuring
    /// `(600, 900)` and `(900, 600)` resolves to one artifact rather than two
    /// copies of the same number.
    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct PortableDiscretizationDistance {
        pub schema_version: u32,
        pub lambda_squared: String,
        /// Lower mode cutoff.
        pub n_modes: usize,
        /// Higher mode cutoff.
        pub m_modes: usize,
        pub precision_bits: u32,
        pub alpha: String,
        /// One entry per requested rule, in the order requested.
        pub measurements: Vec<PortableRuleDistance>,
    }

    /// Measure `D_α(N, M; λ)` and retain it as a `ccm_discretization_distance`
    /// artifact.
    ///
    /// This is the explicit, opt-in capture path for the stabilization
    /// quantity. It cannot be produced by a single-configuration capture
    /// level: `D_α` compares two cutoffs, and an ordinary run resolves one.
    /// `CcmResearchCaptureOptions::maximum` therefore does not and cannot
    /// retain it.
    pub fn capture_ccm_discretization_distance_via_cache(
        first: &crate::ccm::CcmParams,
        second: &crate::ccm::CcmParams,
        cfg: &crate::ccm::hp::HighPrecConfig,
        alpha: &Float,
        rules: &[WeightedIntegrationRule],
        cache: &xc_cache::ArtifactCacheContext<'_>,
    ) -> Result<CcmDiscretizationDistanceHp> {
        use std::collections::BTreeMap;
        use xc_cache::{
            resolve_or_compute_json_artifact_with_dependencies, ArtifactExecutionCacheRequest,
            CacheError, CacheQuality, DependencyRef, SemanticKeyEnvelope, ToolkitVersion,
        };

        // Canonical ascending order, so the symmetric measurement has one
        // identity rather than two.
        let (lower, higher) = if first.n_modes <= second.n_modes {
            (first, second)
        } else {
            (second, first)
        };
        if rules.is_empty() {
            anyhow::bail!("discretization distance requires at least one integration rule");
        }
        for rule in rules {
            rule.validate()?;
        }
        let same_lambda = lower.lambda_sq.is_integer == higher.lambda_sq.is_integer
            && if lower.lambda_sq.is_integer {
                lower.lambda_sq.value_u64 == higher.lambda_sq.value_u64
            } else {
                lower.lambda_sq.value_f64 == higher.lambda_sq.value_f64
            };
        if !same_lambda {
            anyhow::bail!(
                "D_alpha compares two discretizations of one lambda^2; got {} and {}",
                lower.lambda_squared(),
                higher.lambda_squared()
            );
        }
        let lower_canonical =
            crate::ccm::hp::resolve_canonical_even_eigenstate_via_cache(lower, cfg, cache)?;
        let higher_canonical =
            crate::ccm::hp::resolve_canonical_even_eigenstate_via_cache(higher, cfg, cache)?;
        let lower_state = ground_eigenfunction_from_canonical_state(
            lower,
            cfg,
            &lower_canonical.eigenvalue,
            &lower_canonical.eigenvector,
        )?;
        let higher_state = ground_eigenfunction_from_canonical_state(
            higher,
            cfg,
            &higher_canonical.eigenvalue,
            &higher_canonical.eigenvector,
        )?;
        let mut gl_tables = SharedGlTables::new();
        let mut distances = Vec::with_capacity(rules.len());
        for rule in rules {
            distances.push(weighted_alpha_distance_with_tables(
                |u: &Float| lower_state.eigenfunction.eval(u),
                |u: &Float| higher_state.eigenfunction.eval(u),
                &lower_state.lambda,
                alpha,
                *rule,
                cfg.precision_bits,
                Some(&mut gl_tables),
            )?);
        }
        let measurement = CcmDiscretizationDistanceHp {
            lambda_squared: lower.lambda_squared(),
            n_modes: lower.n_modes,
            m_modes: higher.n_modes,
            distances,
        };
        let prec = cfg.precision_bits;
        let lambda_sq_identity = lambda_squared_identity(lower);
        let lower_eigenpair_content_digest = lower_canonical.manifest.content_digest.0.clone();
        let higher_eigenpair_content_digest = higher_canonical.manifest.content_digest.0.clone();
        let eigenpair_dependencies = || {
            let mut dependencies = vec![
                DependencyRef {
                    key: lower_canonical.manifest.key.clone(),
                    content_digest: lower_canonical.manifest.content_digest.clone(),
                    required_quality: CacheQuality::Validated,
                },
                DependencyRef {
                    key: higher_canonical.manifest.key.clone(),
                    content_digest: higher_canonical.manifest.content_digest.clone(),
                    required_quality: CacheQuality::Validated,
                },
            ];
            dependencies.sort_by(|left, right| {
                (
                    left.key.kind.as_str(),
                    left.key.logical_key.as_str(),
                    left.key.parameters_digest.0.as_str(),
                    left.content_digest.0.as_str(),
                )
                    .cmp(&(
                        right.key.kind.as_str(),
                        right.key.logical_key.as_str(),
                        right.key.parameters_digest.0.as_str(),
                        right.content_digest.0.as_str(),
                    ))
            });
            dependencies
        };

        let mut measurements = Vec::with_capacity(rules.len());
        for (rule, distance) in rules.iter().zip(&measurement.distances) {
            measurements.push(PortableRuleDistance {
                rule_family: rule.family().to_owned(),
                quadrature_rule: rule.rule().to_owned(),
                grid_variable: rule.variable().as_str().to_owned(),
                resolution: rule.resolution(),
                distance: decimal(&distance.value, prec),
            });
        }
        let payload = PortableDiscretizationDistance {
            schema_version: 1,
            lambda_squared: lambda_sq_identity.clone(),
            n_modes: lower.n_modes,
            m_modes: higher.n_modes,
            precision_bits: prec,
            alpha: decimal(alpha, prec),
            measurements,
        };

        let semantic_key = SemanticKeyEnvelope {
            schema_version: 1,
            artifact_kind: "ccm_discretization_distance".to_owned(),
            mathematical_semantics_version: "ccm-discretization-distance-v0.14.1-v2".to_owned(),
            resolved_mathematical_parameters: serde_json::json!({
                "lambda_squared": lambda_sq_identity,
                "n_modes": lower.n_modes,
                "m_modes": higher.n_modes,
                "precision_bits": prec,
                "lower_eigenpair_content_digest": lower_eigenpair_content_digest,
                "higher_eigenpair_content_digest": higher_eigenpair_content_digest,
                "alpha": decimal(alpha, prec),
                "rules": rules
                    .iter()
                    .map(|rule| {
                        serde_json::json!({
                            "rule_family": rule.family(),
                            "quadrature_rule": rule.rule(),
                            "grid_variable": rule.variable().as_str(),
                            "resolution": rule.resolution(),
                        })
                    })
                    .collect::<Vec<_>>(),
                "definition": "integral_1^lambda |f_N_lambda(u) - f_M_lambda(u)| u^(-alpha) du"
            }),
            normalization: Some("f(1)=1".to_owned()),
            target: Some("ccm_discretization_distance".to_owned()),
            subspace: Some("even".to_owned()),
            source_data_identities: BTreeMap::new(),
            algorithm_semantics: Some(
                "canonical_eigenpair_uniform_grid_weighted_absolute_difference_v2".to_owned(),
            ),
        };
        let rule_signature = rules
            .iter()
            .map(|rule| {
                format!(
                    "{}-{}-{}",
                    rule.rule(),
                    rule.variable().as_str(),
                    rule.resolution()
                )
            })
            .collect::<Vec<_>>()
            .join("_");
        let logical_key = format!(
            "ccm/discretization-distance/{}/{}/{}/{}/{}",
            lambda_sq_identity, lower.n_modes, higher.n_modes, prec, rule_signature
        );
        let request = ArtifactExecutionCacheRequest {
            operation: "ccm.discretization_distance.resolve_or_compute",
            semantic_key: &semantic_key,
            logical_key: &logical_key,
            resolver: cache.resolver,
            reference_resolver: cache.reference_resolver,
            acceptance: cache.acceptance,
            ordered_overlays: cache.ordered_overlays.clone(),
            mode: cache.mode,
            write_on_miss: cache.write_on_miss,
            write_visibility: cache.write_visibility,
            produced_quality: CacheQuality::Validated,
            producer_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION"))?,
            minimum_reader_version: ToolkitVersion::parse("0.14.1")?,
            maximum_reader_version: None,
            tags: BTreeMap::from([
                ("domain".to_owned(), "ccm".to_owned()),
                ("artifact".to_owned(), "discretization_distance".to_owned()),
            ]),
            provenance_digest: None,
            production_sink: cache.production_sink,
        };
        let resolved = resolve_or_compute_json_artifact_with_dependencies(
            &request,
            || Ok((payload.clone(), eigenpair_dependencies())),
            |artifact| {
                if artifact != &payload {
                    return Err(CacheError::InvalidManifest(
                        "CCM discretization distance does not replay under its stated convention"
                            .to_owned(),
                    ));
                }
                Ok(())
            },
        )?;
        if resolved.value != payload {
            anyhow::bail!("resolved CCM discretization distance disagrees with replayed values");
        }
        Ok(measurement)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "hp")]
    const TEST_TARGET_DIGEST: &str =
        "0000000000000000000000000000000000000000000000000000000000000000";

    fn constant_one_eigenfunction(lambda: f64, n_modes: usize) -> WeilEigenfunctionF64 {
        // Only ξ₀ nonzero: the reconstruction is the constant function 1.
        let mut xi = vec![0.0_f64; 2 * n_modes + 1];
        xi[n_modes] = 2.5; // any nonzero value; normalization removes it
        WeilEigenfunctionF64::from_v_basis(&xi, n_modes, lambda).unwrap()
    }

    /// The named presets are the only place the three derived-capture flags
    /// are set implicitly, so their triples are pinned here. In particular
    /// `MAXIMUM` must not acquire the decomposition: adding a new artifact kind
    /// to a named level breaks `require_reuse` against shards that predate it.
    /// The validator is the only thing standing between a retained payload and
    /// acceptance under a key it does not belong to, so every identity field it
    /// binds is tampered here individually.
    #[cfg(feature = "hp")]
    #[test]
    fn deviation_decomposition_validator_rejects_tampering() {
        use rug::Float;

        const PREC: u32 = 192;
        let steps = 32_usize;
        let target = crate::target::hp::TargetEvaluator::from_environment(PREC).unwrap();
        let mut u_values = Vec::new();
        let mut f_values = Vec::new();
        for k in 0..=steps {
            let t = (k as f64) / (steps as f64);
            let u = Float::with_val(PREC, 1.0 + t * 3.0);
            let mut f = target.value(&u);
            f += target.auxiliary_value(&u).unwrap() * Float::with_val(PREC, 1e-4);
            u_values.push(hp::decimal(&u, PREC));
            f_values.push(hp::decimal(&f, PREC));
        }
        let profile = hp::PortableEigenfunctionProfile {
            schema_version: 1,
            lambda_squared: "16".to_owned(),
            n_modes: 8,
            precision_bits: PREC,
            grid_variable: "uniform_u".to_owned(),
            sample_count: u_values.len(),
            normalization: "f(1)=1".to_owned(),
            u_values,
            f_values,
            normalized_coefficients: Vec::new(),
        };
        let good = hp::compute_deviation_decomposition_payload(&profile, PREC).unwrap();
        let samples = good.sample_count;
        let check = |a: &hp::PortableDeviationDecomposition| {
            hp::structural_deviation_decomposition_check(
                a,
                "16",
                8,
                PREC,
                GridVariable::U,
                samples,
                &good.target_definition_digest,
            )
        };
        assert!(check(&good).is_ok(), "the honest payload must be accepted");

        let mutate = |f: &dyn Fn(&mut hp::PortableDeviationDecomposition)| {
            let mut bad = good.clone();
            f(&mut bad);
            bad
        };
        let cases: Vec<(&str, hp::PortableDeviationDecomposition)> = vec![
            (
                "lambda_squared",
                mutate(&|a| a.lambda_squared = "17".to_owned()),
            ),
            ("n_modes", mutate(&|a| a.n_modes = 9)),
            ("precision_bits", mutate(&|a| a.precision_bits = PREC + 1)),
            ("sample_count", mutate(&|a| a.sample_count += 1)),
            (
                "grid_variable",
                mutate(&|a| a.sampling_grid_variable = "uniform_log_u".to_owned()),
            ),
            (
                "normalization",
                mutate(&|a| a.normalization = "f(0)=1".to_owned()),
            ),
            (
                "target_definition_digest",
                mutate(&|a| a.target_definition_digest = TEST_TARGET_DIGEST.to_owned()),
            ),
            (
                "quadrature_rule",
                mutate(&|a| a.quadrature_rule = "simpson".to_owned()),
            ),
            (
                "sign_convention",
                mutate(&|a| a.sign_convention = "flipped".to_owned()),
            ),
            ("metric_order", mutate(&|a| a.projections.swap(0, 1))),
            (
                "metric_count",
                mutate(&|a| {
                    a.projections.truncate(1);
                }),
            ),
            (
                "negative_norm",
                mutate(&|a| a.projections[0].deviation_norm = "-1".to_owned()),
            ),
            (
                "negative_residual",
                mutate(&|a| a.projections[0].relative_residual = "-0.5".to_owned()),
            ),
            (
                "zero_reference_norm",
                mutate(&|a| a.projections[0].reference_norm = "0".to_owned()),
            ),
            // Schema 2 is the interim pre-release decomposition schema and
            // must be refused, not migrated.
            ("interim_schema_version", mutate(&|a| a.schema_version = 2)),
        ];
        for (name, bad) in cases {
            assert!(
                check(&bad).is_err(),
                "validator accepted a payload tampered in {name}"
            );
        }
    }

    #[cfg(feature = "hp")]
    #[test]
    fn derived_capture_presets_have_the_intended_triples() {
        use hp::DerivedDistanceCapture as D;
        let triple = |c: D| {
            (
                c.resolution_evidence,
                c.residual_analysis,
                c.deviation_decomposition,
            )
        };
        assert_eq!(triple(D::NONE), (false, false, false));
        assert_eq!(triple(D::RESOLUTION_ONLY), (true, false, false));
        assert_eq!(triple(D::RESIDUAL_ONLY), (false, true, false));
        assert_eq!(triple(D::DECOMPOSITION_ONLY), (false, false, true));
        assert_eq!(triple(D::MAXIMUM), (true, true, false));
        assert_eq!(triple(D::MAXIMUM_WITH_DECOMPOSITION), (true, true, true));
    }

    /// A profile whose deviation from the target is an exact multiple of the
    /// reference must return that multiple, in both retained metrics, with nothing
    /// left over. This is the end-to-end check on the artifact payload: it
    /// exercises the retained-decimal round trip, not just the projection.
    #[cfg(feature = "hp")]
    #[test]
    fn deviation_decomposition_recovers_a_planted_reference_amplitude() {
        use rug::Float;

        const PREC: u32 = 192;
        let amplitude = 3.5e-4_f64;
        let lambda = 4.0_f64;
        let steps = 200_usize;

        let target = crate::target::hp::TargetEvaluator::from_environment(PREC).unwrap();
        let mut u_values = Vec::with_capacity(steps + 1);
        let mut f_values = Vec::with_capacity(steps + 1);
        for k in 0..=steps {
            let t = (k as f64) / (steps as f64);
            let u = Float::with_val(PREC, 1.0 + t * (lambda - 1.0));
            let mut f = target.value(&u);
            let mut reference = target.auxiliary_value(&u).unwrap();
            reference *= Float::with_val(PREC, amplitude);
            f += &reference;
            u_values.push(hp::decimal(&u, PREC));
            f_values.push(hp::decimal(&f, PREC));
        }

        let profile = hp::PortableEigenfunctionProfile {
            schema_version: 1,
            lambda_squared: "16".to_owned(),
            n_modes: 8,
            precision_bits: PREC,
            grid_variable: "uniform_u".to_owned(),
            sample_count: u_values.len(),
            normalization: "f(1)=1".to_owned(),
            u_values,
            f_values,
            normalized_coefficients: Vec::new(),
        };

        let payload = hp::compute_deviation_decomposition_payload(&profile, PREC).unwrap();
        assert_eq!(
            payload.projections.len(),
            2,
            "both metrics must be retained"
        );
        assert_eq!(payload.sample_count, steps + 1);

        for projection in &payload.projections {
            let got = Float::with_val(PREC, Float::parse(&projection.amplitude).unwrap());
            let expected = Float::with_val(PREC, amplitude);
            let error = Float::with_val(PREC, &got - &expected).abs() / expected.clone().abs();
            assert!(
                error < Float::with_val(PREC, 1e-25),
                "{}: amplitude {}, expected {amplitude}",
                projection.metric,
                projection.amplitude
            );
            let residual =
                Float::with_val(PREC, Float::parse(&projection.relative_residual).unwrap());
            assert!(
                residual < Float::with_val(PREC, 1e-25),
                "{}: residual {}",
                projection.metric,
                projection.relative_residual
            );
        }

        // The metric identifiers must be distinct, or a consumer cannot tell
        // which reading of the weight produced which amplitude.
        assert_ne!(payload.projections[0].metric, payload.projections[1].metric);
        // The solved parameter must travel with the payload: without it the amplitudes cannot be
        // reproduced from the profile alone.
        let recorded_parameter =
            Float::with_val(PREC, Float::parse(&payload.auxiliary_parameter).unwrap());
        let expected_parameter = target.auxiliary_parameter().unwrap();
        assert!(
            Float::with_val(PREC, &recorded_parameter - &expected_parameter).abs()
                < Float::with_val(PREC, 1e-50),
            "recorded auxiliary parameter {} disagrees with the solved value",
            payload.auxiliary_parameter
        );
        assert!(!payload.quadrature_rule.is_empty());
        assert!(!payload.sign_convention.is_empty());
    }

    /// Normalization is exact at the endpoint and scale-invariant: scaling all
    /// coefficients leaves the normalized eigenfunction unchanged.
    #[test]
    fn eigenfunction_normalization_is_exact_and_scale_invariant() {
        let lambda = 17.0_f64.sqrt();
        let n_modes = 6;
        let mut xi = vec![0.0_f64; 2 * n_modes + 1];
        xi[n_modes] = 1.2;
        for k in 1..=n_modes {
            let value = 0.4 / (k * k) as f64;
            xi[n_modes + k] = value;
            xi[n_modes - k] = value;
        }
        let f = WeilEigenfunctionF64::from_v_basis(&xi, n_modes, lambda).unwrap();
        assert_eq!(f.eval(1.0), 1.0);

        let scaled: Vec<f64> = xi.iter().map(|value| value * -3.75).collect();
        let g = WeilEigenfunctionF64::from_v_basis(&scaled, n_modes, lambda).unwrap();
        for step in 0..=20 {
            let u = 1.0 + (lambda - 1.0) * f64::from(step) / 20.0;
            assert!(
                (f.eval(u) - g.eval(u)).abs() < 1e-12,
                "normalized eigenfunction changed under coefficient scaling at u = {u}"
            );
        }
    }

    /// The self-distance of any profile is exactly zero — the same invariant
    /// the collaboration uses as a harness check ("tested every exported
    /// eigenfunction against itself ... exactly zero").
    #[test]
    fn self_distance_is_exactly_zero() {
        let lambda = 250.0_f64.sqrt();
        let f = constant_one_eigenfunction(lambda, 8);
        let result = weighted_alpha_distance_f64(
            |u| f.eval(u),
            |u| f.eval(u),
            lambda,
            0.5,
            WeightedIntegrationRule::UniformGrid {
                scheme: UniformGridScheme::Trapezoid,
                variable: GridVariable::U,
                steps: 2_000,
            },
        )
        .unwrap();
        assert_eq!(result.value, 0.0);
    }

    /// For the constant profile `f ≡ 1` the weighted norm has the closed form
    /// `∫₁^λ u^{−1/2} du = 2(√λ − 1)`; the computed value must land on it
    /// within the trapezoid error budget, and the result must carry its
    /// convention.
    #[test]
    fn weighted_norm_matches_a_closed_form_and_records_its_convention() {
        let lambda = 16.0;
        let f = constant_one_eigenfunction(lambda, 4);
        let result = weighted_alpha_norm_f64(
            |u| f.eval(u),
            lambda,
            0.5,
            WeightedIntegrationRule::UniformGrid {
                scheme: UniformGridScheme::Trapezoid,
                variable: GridVariable::U,
                steps: 200_000,
            },
        )
        .unwrap();
        let exact = 2.0 * (lambda.sqrt() - 1.0);
        assert!(
            (result.value - exact).abs() < 1e-8,
            "norm {} vs exact {exact}",
            result.value
        );
        assert_eq!(result.alpha, 0.5);
        assert_eq!(result.rule.resolution(), 200_000);
        assert_eq!(result.rule.rule(), "trapezoid");
        assert_eq!(result.rule.variable(), GridVariable::U);
        assert_eq!(result.rule.family(), "uniform_grid");
    }

    /// The convenience wrapper must equal direct integration against the
    /// evaluator compiled from the same private runtime specification.
    #[test]
    fn distance_to_target_matches_direct_runtime_target_integration() {
        let lambda = 20.0;
        let f = constant_one_eigenfunction(lambda, 4);
        let rule = WeightedIntegrationRule::UniformGrid {
            scheme: UniformGridScheme::Trapezoid,
            variable: GridVariable::U,
            steps: 20_000,
        };
        let result = distance_to_target_f64(|u| f.eval(u), lambda, 0.5, rule).unwrap();
        let target = crate::target::TargetEvaluatorF64::from_environment().unwrap();
        let expected =
            weighted_alpha_distance_f64(|u| f.eval(u), |u| target.value(u), lambda, 0.5, rule)
                .unwrap();
        assert_eq!(result, expected);
    }

    /// The retained coefficients round-trip: rebuilding from them reproduces
    /// the eigenfunction exactly. This is what makes a published
    /// `ccm_eigenfunction_profile` usable without the original eigensolve.
    #[test]
    fn normalized_coefficients_round_trip_through_the_reader_constructor() {
        let lambda = 17.0_f64.sqrt();
        let n_modes = 6;
        let mut xi = vec![0.0_f64; 2 * n_modes + 1];
        xi[n_modes] = 1.7;
        for k in 1..=n_modes {
            let value = 0.31 / (k * k) as f64;
            xi[n_modes + k] = value;
            xi[n_modes - k] = value;
        }
        let original = WeilEigenfunctionF64::from_v_basis(&xi, n_modes, lambda).unwrap();

        // Exactly what the profile artifact retains: j = 0..N, normalized.
        let retained = original.normalized_coefficients();

        let rebuilt =
            WeilEigenfunctionF64::from_normalized_coefficients(&retained, lambda).unwrap();
        assert_eq!(rebuilt.eval(1.0), 1.0);
        for step in 0..=40 {
            let u = 1.0 + (lambda - 1.0) * f64::from(step) / 40.0;
            assert!(
                (original.eval(u) - rebuilt.eval(u)).abs() < 1e-12,
                "rebuilt eigenfunction differs at u = {u}"
            );
        }
        assert!(WeilEigenfunctionF64::from_normalized_coefficients(&[], lambda).is_err());
    }

    /// A profile that never crosses the target reports no sign change, and one
    /// that crosses reports a bracket containing it. The count is what decides
    /// whether Gauss-Legendre keeps its spectral advantage for a
    /// configuration, so it must not be inferred or assumed.
    #[test]
    fn target_crossings_detect_sign_changes_and_bracket_them() {
        let lambda = 4.0_f64;

        // The benign test target decreases from one, so f == 1 stays weakly above it and the
        // difference never changes sign.
        let none = target_crossings_f64(|_| 1.0, lambda, 400, GridVariable::U).unwrap();
        assert_eq!(none.crossings(), 0);
        assert!(none.integrand_appears_smooth());
        assert_eq!(none.initial_sign, 1);

        // Cross deliberately: start below the target near u = 1, end above it.
        let crossing =
            target_crossings_f64(|u| 0.5 * (u - 1.0), lambda, 400, GridVariable::U).unwrap();
        assert!(
            crossing.crossings() >= 1,
            "a deliberate crossing must be detected"
        );
        assert!(!crossing.integrand_appears_smooth());
        for (left, right) in &crossing.brackets {
            assert!(*left >= 1.0 && right <= &lambda && left < right);
        }
        assert!(target_crossings_f64(|_| 1.0, lambda, 1, GridVariable::U).is_err());
        assert!(target_crossings_f64(|_| 1.0, 1.0, 100, GridVariable::U).is_err());
    }

    #[test]
    fn invalid_eigenfunctions_are_rejected() {
        // Wrong length.
        assert!(WeilEigenfunctionF64::from_v_basis(&[1.0; 4], 2, 4.0).is_err());
        // λ ≤ 1.
        assert!(WeilEigenfunctionF64::from_v_basis(&[1.0; 5], 2, 1.0).is_err());
        // Un-normalizable: f_raw(1) = 0 for this coefficient choice
        // (ξ₀ = 0, ξ₁ mirrored evenly cancels at u = 1 only if ξ₀ + 2Σ(−1)ⁿξₙ = 0).
        let mut xi = vec![0.0_f64; 5];
        xi[2] = 2.0; // ξ₀
        xi[3] = 1.0; // ξ₁ contributes 2·(−1)·1 = −2 at u = 1
        assert!(WeilEigenfunctionF64::from_v_basis(&xi, 2, 4.0).is_err());
    }

    #[cfg(feature = "hp")]
    mod bench_tests {
        use super::super::hp::WeilEigenfunction;
        use rug::Float;
        use std::time::Instant;

        const PREC: u32 = 3392;
        const POINTS: usize = 200;

        /// Accumulated `eval` over a fixed grid, at 60 decimal digits.
        ///
        /// Captured from the serial `raw_eval` this parallel implementation
        /// replaced. `raw_eval` now evaluates its coefficient terms in
        /// parallel and folds them in coefficient order; these values pin that
        /// the fold order was preserved, because a parallel *reduction* would
        /// re-associate the additions and move the low bits, silently breaking
        /// replay of every retained distance artifact.
        const SERIAL_REFERENCE: [(usize, &str); 3] = [
            (
                120,
                "137.708585009846794326000907520096899231310981598310101995016",
            ),
            (
                500,
                "48.6923970427694007923006927244400122608751183626573040197533",
            ),
            (
                1500,
                "236.486064997542093149807369759307932838774852903413564909677",
            ),
        ];

        /// Deterministic non-trivial coefficients; shared by both tests below.
        fn probe(n_modes: usize) -> Option<(WeilEigenfunction, Float, Float)> {
            let lambda = Float::with_val(PREC, 13u32).sqrt();
            let mut xi = Vec::with_capacity(2 * n_modes + 1);
            for k in 0..(2 * n_modes + 1) {
                let mut v = Float::with_val(PREC, (k as i64 % 17) - 8);
                v /= Float::with_val(PREC, (k + 3) as u32);
                xi.push(v);
            }
            let ef = WeilEigenfunction::from_v_basis(&xi, n_modes, &lambda, PREC).ok()?;
            let step = Float::with_val(PREC, &lambda - Float::with_val(PREC, 1u32)) / POINTS as u32;
            Some((ef, step, lambda))
        }

        fn accumulate(ef: &WeilEigenfunction, step: &Float) -> Float {
            let mut sink = Float::with_val(PREC, 0u32);
            for i in 0..POINTS {
                let mut u = step.clone();
                u *= i as u32;
                u += Float::with_val(PREC, 1u32);
                sink += ef.eval(&u);
            }
            sink
        }

        /// The parallel coefficient fold must reproduce the serial reference
        /// exactly. This is the guard on artifact replay, not a tolerance test.
        #[test]
        fn raw_eval_is_bit_identical_to_the_serial_reference() {
            for (n_modes, expected) in SERIAL_REFERENCE {
                let Some((ef, step, _)) = probe(n_modes) else {
                    panic!("probe eigenfunction at N={n_modes} could not be constructed");
                };
                let got = accumulate(&ef, &step).to_string_radix(10, Some(60));
                assert_eq!(
                    got, expected,
                    "N={n_modes}: parallel raw_eval diverged from the serial reference"
                );
            }
        }

        /// Timing probe for the hot path. Not an assertion of speed - run it
        /// explicitly on the target machine:
        ///   cargo test --release -p xc-spectral --features arb \
        ///     bench_raw_eval -- --ignored --nocapture
        #[test]
        #[ignore = "timing probe; run explicitly with --nocapture"]
        fn bench_raw_eval_hot_path() {
            for (n_modes, _) in SERIAL_REFERENCE {
                let Some((ef, step, _)) = probe(n_modes) else {
                    continue;
                };
                let start = Instant::now();
                let sink = accumulate(&ef, &step);
                let elapsed = start.elapsed();
                println!(
                    "N={n_modes:>5}  {POINTS} evals  {:>9.3} ms  ({:>7.3} ms/eval)  sink={}",
                    elapsed.as_secs_f64() * 1e3,
                    elapsed.as_secs_f64() * 1e3 / POINTS as f64,
                    sink.to_string_radix(10, Some(60))
                );
            }
        }
    }

    #[cfg(feature = "hp")]
    mod hp_tests {
        use super::super::hp;
        use super::super::WeightedIntegrationRule;
        use super::TEST_TARGET_DIGEST;
        use rug::Float;
        use xc_numerics::grid_integral::{GridVariable, UniformGridScheme};

        fn test_target_digest() -> String {
            crate::target::TargetProfileSpec::from_environment()
                .unwrap()
                .digest()
                .unwrap()
        }

        fn hp_constant_one(lambda: &Float, n_modes: usize, prec: u32) -> hp::WeilEigenfunction {
            let mut xi: Vec<Float> = (0..(2 * n_modes + 1))
                .map(|_| Float::with_val(prec, 0u32))
                .collect();
            xi[n_modes] = Float::with_val(prec, 3u32);
            hp::WeilEigenfunction::from_v_basis(&xi, n_modes, lambda, prec).unwrap()
        }

        /// HP self-distance is exactly zero at every precision — mirrors the
        /// collaboration's own harness invariant at 3535–7189 bits.
        #[test]
        fn hp_self_distance_is_exactly_zero() {
            for prec in [256u32, 1024] {
                let lambda = Float::with_val(prec, 250u32).sqrt();
                let f = hp_constant_one(&lambda, 6, prec);
                let alpha = Float::with_val(prec, 0.5);
                let result = hp::weighted_alpha_distance(
                    |u: &Float| f.eval(u),
                    |u: &Float| f.eval(u),
                    &lambda,
                    &alpha,
                    WeightedIntegrationRule::UniformGrid {
                        scheme: UniformGridScheme::Trapezoid,
                        variable: GridVariable::U,
                        steps: 500,
                    },
                    prec,
                )
                .unwrap();
                assert_eq!(result.value, 0u32, "nonzero self-distance at {prec} bits");
            }
        }

        /// The HP weighted norm of the constant profile lands on the closed
        /// form `2(√λ − 1)` within the trapezoid budget.
        #[test]
        fn hp_weighted_norm_matches_the_closed_form() {
            let prec = 256;
            let lambda = Float::with_val(prec, 16u32);
            let f = hp_constant_one(&lambda, 4, prec);
            let alpha = Float::with_val(prec, 0.5);
            let result = hp::weighted_alpha_norm(
                |u: &Float| f.eval(u),
                &lambda,
                &alpha,
                WeightedIntegrationRule::UniformGrid {
                    scheme: UniformGridScheme::Trapezoid,
                    variable: GridVariable::U,
                    steps: 20_000,
                },
                prec,
            )
            .unwrap();
            let exact = (Float::with_val(prec, 16u32).sqrt() - 1u32) * 2u32;
            let error = Float::with_val(prec, &result.value - &exact).abs();
            // The trapezoid Euler-Maclaurin leading term for this integrand is
            // h^2/12 * (F'(16) - F'(1)) ~ 2.31e-8 at 20,000 steps; the
            // computed error must sit at that budget, not above it.
            assert!(error < 5e-8, "error {error:?}");
        }

        /// Shared Gauss--Legendre tables are payload-preserving: the value
        /// computed through a shared table is bit-identical to the per-call
        /// construction path, and one table serves repeated measurements.
        #[test]
        fn shared_gl_tables_preserve_values_exactly() {
            let prec = 256;
            let lambda = Float::with_val(prec, 13u32).sqrt();
            let f = hp_constant_one(&lambda, 6, prec);
            let alpha = Float::with_val(prec, 0.5);
            let rule = WeightedIntegrationRule::GaussLegendre {
                points: 64,
                variable: GridVariable::U,
            };
            let per_call =
                hp::weighted_alpha_norm(|u: &Float| f.eval(u), &lambda, &alpha, rule, prec)
                    .unwrap();
            let mut tables = hp::SharedGlTables::new();
            for _ in 0..3 {
                let shared = hp::weighted_alpha_norm_with_tables(
                    |u: &Float| f.eval(u),
                    &lambda,
                    &alpha,
                    rule,
                    prec,
                    Some(&mut tables),
                )
                .unwrap();
                assert_eq!(
                    shared.value, per_call.value,
                    "shared-table value differs from per-call construction"
                );
            }
        }

        /// Managed quadrature reuse is exactly the cache-off arithmetic route:
        /// a cold artifact and a required warm reuse must both produce the
        /// identical distance-layer value.
        #[test]
        fn managed_gl_tables_preserve_cache_off_values_exactly() {
            use xc_cache::{
                ArtifactCacheContext, ArtifactExecutionCacheMode, CacheLayer, CachePolicy,
                CacheQuality, CacheResolver, CacheVisibility, CertificationFailurePolicy,
                FilesystemCacheStore, ToolkitVersion,
            };

            let root = std::env::temp_dir().join(format!(
                "xc-spectral-managed-gl-exact-{}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&root);
            let resolver = CacheResolver::new(vec![CacheLayer {
                precedence: 0,
                store: Box::new(FilesystemCacheStore::new(
                    "workstation",
                    &root,
                    true,
                    CacheVisibility::Local,
                )),
            }]);
            let policy = CachePolicy {
                current_toolkit_version: ToolkitVersion::parse("0.14.1").unwrap(),
                minimum_quality: CacheQuality::Validated,
                accepted_schema_versions: vec![1],
                allow_deprecated: false,
                allow_quarantined: false,
                allowed_visibilities: vec![CacheVisibility::Local],
            };
            let context = |mode, write_on_miss| ArtifactCacheContext {
                resolver: Some(&resolver),
                reference_resolver: None,
                acceptance: Some(&policy),
                ordered_overlays: vec!["workstation".to_owned()],
                mode,
                write_on_miss,
                write_visibility: CacheVisibility::Local,
                requested_assurance: xc_core::AssuranceLevel::Computed,
                certification_failure_policy: CertificationFailurePolicy::RetainComputedFailRun,
                production_sink: None,
            };

            let prec = 192;
            let lambda = Float::with_val(prec, 13u32).sqrt();
            let f = hp_constant_one(&lambda, 6, prec);
            let alpha = Float::with_val(prec, 0.5);
            let rule = WeightedIntegrationRule::GaussLegendre {
                points: 32,
                variable: GridVariable::U,
            };
            let cache_off =
                hp::weighted_alpha_norm(|u: &Float| f.eval(u), &lambda, &alpha, rule, prec)
                    .unwrap();

            let mut cold_tables = hp::SharedGlTables::new();
            cold_tables
                .preload_managed(
                    &[rule],
                    prec,
                    &context(ArtifactExecutionCacheMode::PreferReuse, true),
                )
                .unwrap();
            let cold = hp::weighted_alpha_norm_with_tables(
                |u: &Float| f.eval(u),
                &lambda,
                &alpha,
                rule,
                prec,
                Some(&mut cold_tables),
            )
            .unwrap();

            let mut warm_tables = hp::SharedGlTables::new();
            warm_tables
                .preload_managed(
                    &[rule],
                    prec,
                    &context(ArtifactExecutionCacheMode::RequireReuse, false),
                )
                .unwrap();
            let warm = hp::weighted_alpha_norm_with_tables(
                |u: &Float| f.eval(u),
                &lambda,
                &alpha,
                rule,
                prec,
                Some(&mut warm_tables),
            )
            .unwrap();

            assert_eq!(cold.value, cache_off.value);
            assert_eq!(warm.value, cache_off.value);
            std::fs::remove_dir_all(root).unwrap();
        }

        /// Trapezoid refinement reuses only exactly identical nested nodes.
        /// The numerical result stays bit-identical while Q/2Q/4Q needs only
        /// the newly introduced abscissae after the Q samples are seeded.
        #[test]
        fn nested_refinement_reuses_exact_values_without_changing_results() {
            let prec = 192;
            let lambda = Float::with_val(prec, 13u32).sqrt();
            let eigenfunction = hp_constant_one(&lambda, 6, prec);
            let alpha = Float::with_val(prec, 0.5);

            for variable in [GridVariable::U, GridVariable::LogU] {
                let base_rule = WeightedIntegrationRule::UniformGrid {
                    scheme: UniformGridScheme::Trapezoid,
                    variable,
                    steps: 8,
                };
                let samples = std::cell::RefCell::new(Vec::new());
                hp::distance_to_target(
                    |u: &Float| {
                        let value = eigenfunction.eval(u);
                        samples.borrow_mut().push((u.clone(), value.clone()));
                        value
                    },
                    &lambda,
                    &alpha,
                    base_rule,
                    prec,
                )
                .unwrap();
                let samples = samples.into_inner();
                assert_eq!(samples.len(), 9);
                // Q, 2Q and 4Q abscissae are enumerated and evaluated once.
                // Every Q point recurs in 2Q and every 2Q point in 4Q, so the
                // union is exactly the 4Q set: 33 abscissae covering all three
                // levels instead of 9 + 17 + 33 = 59 evaluations.
                let exact_values = hp::PrecomputedEigenfunctionValues::build(
                    &eigenfunction,
                    Some(&samples),
                    &[
                        (UniformGridScheme::Trapezoid, variable, 16),
                        (UniformGridScheme::Trapezoid, variable, 32),
                    ],
                    &lambda,
                    prec,
                );
                assert_eq!(
                    exact_values.retained(),
                    33,
                    "nested grids should collapse to the 4Q abscissa set"
                );

                let twice_rule = WeightedIntegrationRule::UniformGrid {
                    scheme: UniformGridScheme::Trapezoid,
                    variable,
                    steps: 16,
                };
                let direct_twice = hp::distance_to_target(
                    |u: &Float| eigenfunction.eval(u),
                    &lambda,
                    &alpha,
                    twice_rule,
                    prec,
                )
                .unwrap();
                let reused_twice = hp::distance_to_target(
                    |u: &Float| exact_values.eval(u),
                    &lambda,
                    &alpha,
                    twice_rule,
                    prec,
                )
                .unwrap();
                assert_eq!(reused_twice.value, direct_twice.value);

                let four_rule = WeightedIntegrationRule::UniformGrid {
                    scheme: UniformGridScheme::Trapezoid,
                    variable,
                    steps: 32,
                };
                let direct_four = hp::distance_to_target(
                    |u: &Float| eigenfunction.eval(u),
                    &lambda,
                    &alpha,
                    four_rule,
                    prec,
                )
                .unwrap();
                let reused_four = hp::distance_to_target(
                    |u: &Float| exact_values.eval(u),
                    &lambda,
                    &alpha,
                    four_rule,
                    prec,
                )
                .unwrap();
                assert_eq!(reused_four.value, direct_four.value);
            }
        }

        /// A zero-resolution rule fails before any eigenstate or quadrature
        /// work, with a diagnostic naming the defect.
        #[test]
        fn zero_resolution_rules_fail_fast() {
            let prec = 256;
            let lambda = Float::with_val(prec, 13u32).sqrt();
            let f = hp_constant_one(&lambda, 6, prec);
            let alpha = Float::with_val(prec, 0.5);
            let error = hp::weighted_alpha_distance(
                |u: &Float| f.eval(u),
                |u: &Float| f.eval(u),
                &lambda,
                &alpha,
                WeightedIntegrationRule::UniformGrid {
                    scheme: UniformGridScheme::Trapezoid,
                    variable: GridVariable::U,
                    steps: 0,
                },
                prec,
            )
            .unwrap_err();
            assert!(
                error.to_string().contains("positive resolution"),
                "unexpected diagnostic: {error}"
            );
        }

        /// Reuse-mode validation must bind the retained payload to the exact
        /// requested convention. A content-valid artifact with a substituted
        /// rule, alpha, vector shape, or non-finite scalar is not a cache hit
        /// for this request.
        #[test]
        fn retained_distance_payloads_are_fully_bound_to_the_request() {
            let prec = 128;
            let alpha = Float::with_val(prec, 0.5);
            let rule = WeightedIntegrationRule::UniformGrid {
                scheme: UniformGridScheme::Trapezoid,
                variable: GridVariable::U,
                steps: 4_000,
            };
            let profile = hp::PortableEigenfunctionProfile {
                schema_version: 1,
                lambda_squared: "4".to_owned(),
                n_modes: 2,
                precision_bits: prec,
                grid_variable: GridVariable::U.as_str().to_owned(),
                sample_count: 2,
                normalization: "f(1)=1".to_owned(),
                u_values: vec!["1".to_owned(), "2".to_owned()],
                f_values: vec!["1".to_owned(), "0.5".to_owned()],
                normalized_coefficients: vec![
                    "0.5".to_owned(),
                    "-0.25".to_owned(),
                    "0.125".to_owned(),
                ],
            };
            hp::validate_portable_eigenfunction_profile(&profile, "4", 2, prec, GridVariable::U, 1)
                .unwrap();

            let mut malformed_profile = profile.clone();
            malformed_profile.normalized_coefficients.pop();
            assert!(hp::validate_portable_eigenfunction_profile(
                &malformed_profile,
                "4",
                2,
                prec,
                GridVariable::U,
                1,
            )
            .is_err());
            let mut unordered_profile = profile.clone();
            unordered_profile.u_values.swap(0, 1);
            assert!(hp::validate_portable_eigenfunction_profile(
                &unordered_profile,
                "4",
                2,
                prec,
                GridVariable::U,
                1,
            )
            .is_err());

            let expected_eigenvalue = Float::with_val(
                prec,
                Float::parse("1e-20").expect("fixed eigenvalue is valid"),
            );
            let expected_target_digest = test_target_digest();
            let validation_request = || hp::TargetDistanceValidationRequest {
                target_definition_digest: &expected_target_digest,
                lambda_squared: "4",
                n_modes: 2,
                precision_bits: prec,
                alpha: &alpha,
                rules: std::slice::from_ref(&rule),
                expected_eigenvalue: &expected_eigenvalue,
            };
            let distance = hp::PortableTargetDistance {
                schema_version: 2,
                target_definition_digest: expected_target_digest.clone(),
                lambda_squared: "4".to_owned(),
                n_modes: 2,
                precision_bits: prec,
                alpha: hp::decimal(&alpha, prec),
                measurements: vec![hp::PortableRuleMeasurement {
                    rule_family: rule.family().to_owned(),
                    quadrature_rule: rule.rule().to_owned(),
                    grid_variable: rule.variable().as_str().to_owned(),
                    resolution: rule.resolution(),
                    distance_to_target: "0.01".to_owned(),
                    eigenfunction_norm: "0.5".to_owned(),
                }],
                eigenvalue: hp::decimal(&expected_eigenvalue, prec),
            };
            hp::validate_portable_target_distance(&distance, validation_request()).unwrap();

            let mut substituted_rule = distance.clone();
            substituted_rule.measurements[0].quadrature_rule = "midpoint".to_owned();
            assert!(
                hp::validate_portable_target_distance(&substituted_rule, validation_request(),)
                    .is_err()
            );
            let mut substituted_alpha = distance.clone();
            substituted_alpha.alpha = "0.25".to_owned();
            assert!(hp::validate_portable_target_distance(
                &substituted_alpha,
                validation_request(),
            )
            .is_err());
            let mut substituted_eigenvalue = distance.clone();
            substituted_eigenvalue.eigenvalue = "-1e-20".to_owned();
            assert!(hp::validate_portable_target_distance(
                &substituted_eigenvalue,
                validation_request(),
            )
            .is_err());
            let mut substituted_target = distance.clone();
            substituted_target.target_definition_digest = TEST_TARGET_DIGEST.to_owned();
            assert!(hp::validate_portable_target_distance(
                &substituted_target,
                validation_request(),
            )
            .is_err());
            let mut non_finite = distance;
            non_finite.measurements[0].distance_to_target = "NaN".to_owned();
            assert!(
                hp::validate_portable_target_distance(&non_finite, validation_request(),).is_err()
            );
        }

        /// Resolution evidence is computed from the lossless coefficients and
        /// refines only like-for-like uniform grids. The independent
        /// Gauss--Legendre cross-check remains in the target-distance artifact
        /// and must not be silently converted into a second refinement series.
        #[test]
        fn distance_resolution_evidence_follows_its_fixed_policy() {
            let prec = 192;
            let lambda = Float::with_val(prec, 3u32);
            let alpha = Float::with_val(prec, 0.5);
            let eigenfunction = hp_constant_one(&lambda, 4, prec);
            let rules = [
                WeightedIntegrationRule::UniformGrid {
                    scheme: UniformGridScheme::Trapezoid,
                    variable: GridVariable::U,
                    steps: 8,
                },
                WeightedIntegrationRule::GaussLegendre {
                    points: 8,
                    variable: GridVariable::U,
                },
            ];
            let mut base_value_samples = Vec::with_capacity(rules.len());
            let base_distances = rules
                .iter()
                .map(|rule| {
                    let samples = std::cell::RefCell::new(Vec::new());
                    let distance = hp::distance_to_target(
                        |u: &Float| {
                            let value = eigenfunction.eval(u);
                            if matches!(*rule, WeightedIntegrationRule::UniformGrid { .. }) {
                                samples.borrow_mut().push((u.clone(), value.clone()));
                            }
                            value
                        },
                        &lambda,
                        &alpha,
                        *rule,
                        prec,
                    )
                    .unwrap();
                    base_value_samples.push(samples.into_inner());
                    distance
                })
                .collect::<Vec<_>>();
            let baseline = hp::compute_distance_resolution_evidence(
                &eigenfunction,
                &lambda,
                &alpha,
                &rules,
                &base_distances,
                &test_target_digest(),
                "9",
                prec,
            )
            .unwrap();
            let evidence = hp::compute_distance_resolution_evidence_with_samples(
                &eigenfunction,
                &lambda,
                &alpha,
                &rules,
                hp::ResolutionEvidenceEvaluationSource {
                    base_distances: &base_distances,
                    base_value_samples: Some(&base_value_samples),
                },
                &test_target_digest(),
                "9",
                prec,
            )
            .unwrap();
            assert_eq!(evidence, baseline);

            assert_eq!(evidence.coefficient_count, 5);
            assert_eq!(
                evidence
                    .coefficient_tail
                    .iter()
                    .map(|tail| tail.threshold.as_str())
                    .collect::<Vec<_>>(),
                ["1e-15", "1e-30", "1e-45"]
            );
            assert!(evidence
                .coefficient_tail
                .iter()
                .all(|tail| tail.effective_bandwidth == Some(0)));
            assert_eq!(evidence.refinements.len(), 1);
            let refinement = &evidence.refinements[0];
            assert_eq!(refinement.rule_family, "uniform_grid");
            assert_eq!(refinement.base_resolution, 8);
            assert_eq!(refinement.twice_resolution, 16);
            assert_eq!(refinement.four_times_resolution, Some(32));
            assert_eq!(refinement.final_resolution, 32);
            hp::validate_portable_distance_resolution_evidence(
                &evidence,
                &test_target_digest(),
                "9",
                4,
                prec,
                &alpha,
                &rules,
            )
            .unwrap();

            let mut tampered = evidence;
            tampered.refinements[0].final_resolution = 16;
            assert!(hp::validate_portable_distance_resolution_evidence(
                &tampered,
                &test_target_digest(),
                "9",
                4,
                prec,
                &alpha,
                &rules,
            )
            .is_err());
        }

        /// A two-sided residual must survive its own validator.
        ///
        /// Regression for a defect that made `ccm_target_residual_analysis`
        /// impossible to retain. `decimal` emits `precision_bits * log10(2)`
        /// digits, two fewer than an exact round trip needs, so re-parsing a
        /// retained mass does not recover the value that produced it. The
        /// one-sided masses were derived from the unrounded working-precision
        /// values while the reader re-derives them from the retained decimals
        /// and demands exact string equality, so every payload was rejected.
        ///
        /// The pre-existing coverage could not catch this: it used a constant
        /// eigenfunction whose residual never changes sign, so the snap made
        /// `signed == absolute`, `negative` was exactly `"0"`, and the halving
        /// round-tripped trivially. This case keeps both one-sided masses
        /// genuinely nonzero, which is where the lost digits show up.
        #[test]
        fn two_sided_residual_masses_survive_their_own_validator() {
            let prec = 192;
            let lambda = Float::with_val(prec, 3u32);
            let alpha = Float::with_val(prec, 0.5);

            // Equal nonzero cosine coefficients give a strongly oscillating
            // normalized profile that crosses the benign runtime test target.
            let n_modes = 4usize;
            let xi = vec![Float::with_val(prec, 1u32); 2 * n_modes + 1];
            let eigenfunction =
                hp::WeilEigenfunction::from_v_basis(&xi, n_modes, &lambda, prec).unwrap();

            let rules = [WeightedIntegrationRule::UniformGrid {
                scheme: UniformGridScheme::Trapezoid,
                variable: GridVariable::U,
                steps: 64,
            }];
            let base_distances = rules
                .iter()
                .map(|rule| {
                    hp::distance_to_target(
                        |u: &Float| eigenfunction.eval(u),
                        &lambda,
                        &alpha,
                        *rule,
                        prec,
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>();
            let u_values = (0..=16)
                .map(|index| {
                    let mut u = Float::with_val(prec, &lambda - 1u32);
                    u *= index;
                    u /= 16u32;
                    u += 1u32;
                    u
                })
                .collect::<Vec<_>>();
            let f_values = u_values
                .iter()
                .map(|u| eigenfunction.eval(u))
                .collect::<Vec<_>>();

            let analysis = hp::compute_target_residual_analysis(hp::TargetResidualAnalysisSource {
                eigenfunction: &eigenfunction,
                lambda: &lambda,
                alpha: &alpha,
                rules: &rules,
                base_distances: &base_distances,
                precomputed_signed_residuals: None,
                target_definition_digest: &test_target_digest(),
                lambda_squared: "9",
                sampling_variable: GridVariable::U,
                precision_bits: prec,
                u_values: &u_values,
                f_values: &f_values,
            })
            .unwrap();

            // The case is only meaningful if the residual really is two-sided.
            let measurement = &analysis.measurements[0];
            assert_ne!(
                measurement.negative_residual_mass, "0",
                "probe profile does not cross the target; the regression would not bite"
            );
            assert_ne!(
                measurement.positive_residual_mass, "0",
                "probe profile does not cross the target; the regression would not bite"
            );
            assert_ne!(
                measurement.absolute_residual_mass, measurement.signed_residual_mass,
                "a snapped one-sided residual would not exercise the halving"
            );

            hp::validate_portable_target_residual_analysis(
                &analysis,
                hp::TargetResidualAnalysisValidationRequest {
                    target_definition_digest: &test_target_digest(),
                    lambda_squared: "9",
                    n_modes,
                    precision_bits: prec,
                    alpha: &alpha,
                    rules: &rules,
                    variable: GridVariable::U,
                    profile_steps: 16,
                },
            )
            .expect("a freshly computed residual analysis must satisfy its own validator");
        }

        /// The residual artifact exposes the directional information hidden
        /// by absolute distance while preserving that established value.
        #[test]
        fn target_residual_analysis_records_one_sided_mass_and_sign_structure() {
            let prec = 192;
            let lambda = Float::with_val(prec, 3u32);
            let alpha = Float::with_val(prec, 0.5);
            let eigenfunction = hp_constant_one(&lambda, 4, prec);
            let rules = [WeightedIntegrationRule::UniformGrid {
                scheme: UniformGridScheme::Trapezoid,
                variable: GridVariable::U,
                steps: 64,
            }];
            let base_distances = rules
                .iter()
                .map(|rule| {
                    hp::distance_to_target(
                        |u: &Float| eigenfunction.eval(u),
                        &lambda,
                        &alpha,
                        *rule,
                        prec,
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>();
            let u_values = (0..=16)
                .map(|index| {
                    let mut u = Float::with_val(prec, &lambda - 1u32);
                    u *= index;
                    u /= 16u32;
                    u += 1u32;
                    u
                })
                .collect::<Vec<_>>();
            let f_values = u_values
                .iter()
                .map(|u| eigenfunction.eval(u))
                .collect::<Vec<_>>();
            let analysis = hp::compute_target_residual_analysis(hp::TargetResidualAnalysisSource {
                eigenfunction: &eigenfunction,
                lambda: &lambda,
                alpha: &alpha,
                rules: &rules,
                base_distances: &base_distances,
                u_values: &u_values,
                f_values: &f_values,
                precomputed_signed_residuals: None,
                target_definition_digest: &test_target_digest(),
                lambda_squared: "9",
                sampling_variable: GridVariable::U,
                precision_bits: prec,
            })
            .unwrap();

            assert!(analysis.crossing_brackets.is_empty());
            assert!(analysis.sample_signs.iter().all(|sign| *sign >= 0));
            assert_eq!(analysis.measurements.len(), 1);
            assert_eq!(
                analysis.measurements[0].absolute_residual_mass,
                analysis.measurements[0].signed_residual_mass
            );
            assert_eq!(analysis.measurements[0].negative_residual_mass, "0");
            hp::validate_portable_target_residual_analysis(
                &analysis,
                hp::TargetResidualAnalysisValidationRequest {
                    target_definition_digest: &test_target_digest(),
                    lambda_squared: "9",
                    n_modes: 4,
                    precision_bits: prec,
                    alpha: &alpha,
                    rules: &rules,
                    variable: GridVariable::U,
                    profile_steps: 16,
                },
            )
            .unwrap();

            let mut tampered = analysis;
            tampered.sample_signs[1] = 2;
            assert!(hp::validate_portable_target_residual_analysis(
                &tampered,
                hp::TargetResidualAnalysisValidationRequest {
                    target_definition_digest: &test_target_digest(),
                    lambda_squared: "9",
                    n_modes: 4,
                    precision_bits: prec,
                    alpha: &alpha,
                    rules: &rules,
                    variable: GridVariable::U,
                    profile_steps: 16,
                },
            )
            .is_err());
        }

        /// A real managed capture writes the canonical-eigenpair-bound profile
        /// and distance, then backfills both diagnostic artifacts from those
        /// retained values. Required reuse subsequently reproduces the same
        /// canonical state without a second numerical route.
        #[test]
        fn managed_distance_analysis_backfills_and_round_trips_from_cache() {
            use xc_cache::{
                ArtifactCacheContext, ArtifactExecutionCacheMode, CacheLayer, CachePolicy,
                CacheQuality, CacheResolver, CacheVisibility, CertificationFailurePolicy,
                FilesystemCacheStore, ToolkitVersion,
            };

            let root = std::env::temp_dir().join(format!(
                "xc-spectral-distance-analysis-backfill-{}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&root);
            let resolver = CacheResolver::new(vec![CacheLayer {
                precedence: 0,
                store: Box::new(FilesystemCacheStore::new(
                    "workstation",
                    &root,
                    true,
                    CacheVisibility::Local,
                )),
            }]);
            let policy = CachePolicy {
                current_toolkit_version: ToolkitVersion::parse("0.14.1").unwrap(),
                minimum_quality: CacheQuality::Validated,
                accepted_schema_versions: vec![1],
                allow_deprecated: false,
                allow_quarantined: false,
                allowed_visibilities: vec![CacheVisibility::Local],
            };
            let context = |mode, write_on_miss| ArtifactCacheContext {
                resolver: Some(&resolver),
                reference_resolver: None,
                acceptance: Some(&policy),
                ordered_overlays: vec!["workstation".to_owned()],
                mode,
                write_on_miss,
                write_visibility: CacheVisibility::Local,
                requested_assurance: xc_core::AssuranceLevel::Computed,
                certification_failure_policy: CertificationFailurePolicy::RetainComputedFailRun,
                production_sink: None,
            };
            let params = crate::ccm::CcmParams::from_lambda_sq_integer(5, 2);
            let cfg = crate::ccm::hp::HighPrecConfig::for_decimal_digits(40);
            let alpha = Float::with_val(cfg.precision_bits, 0.5);
            let rules = [
                WeightedIntegrationRule::UniformGrid {
                    scheme: UniformGridScheme::Trapezoid,
                    variable: GridVariable::U,
                    steps: 32,
                },
                WeightedIntegrationRule::GaussLegendre {
                    points: 16,
                    variable: GridVariable::U,
                },
            ];
            let first = hp::capture_ccm_distance_via_cache(
                &params,
                &cfg,
                &alpha,
                &rules,
                16,
                &context(ArtifactExecutionCacheMode::PreferReuse, true),
            )
            .unwrap();
            let primary = crate::ccm::hp::resolve_canonical_even_eigenstate_via_cache(
                &params,
                &cfg,
                &context(ArtifactExecutionCacheMode::RequireReuse, false),
            )
            .unwrap();
            assert_eq!(
                first.eigenvalue, primary.eigenvalue,
                "distance capture must retain the exact canonical claim eigenvalue"
            );
            let backfilled = hp::capture_ccm_distance_with_numerical_analysis_via_cache(
                &params,
                &cfg,
                &alpha,
                &rules,
                16,
                &context(ArtifactExecutionCacheMode::PreferReuse, true),
            )
            .unwrap();
            let refreshed = hp::capture_ccm_distance_with_numerical_analysis_via_cache(
                &params,
                &cfg,
                &alpha,
                &rules,
                16,
                &context(ArtifactExecutionCacheMode::Refresh, true),
            )
            .unwrap();
            let reused = hp::capture_ccm_distance_with_numerical_analysis_via_cache(
                &params,
                &cfg,
                &alpha,
                &rules,
                16,
                &context(ArtifactExecutionCacheMode::RequireReuse, false),
            )
            .unwrap();
            assert_eq!(first.distances.len(), 2);
            let replay_tolerance = Float::with_val(
                cfg.precision_bits,
                Float::parse("1e-55").expect("fixed replay tolerance is valid"),
            );
            let mut replay_fields = vec![(
                "eigenvalue".to_owned(),
                &backfilled.eigenvalue,
                &reused.eigenvalue,
            )];
            let refresh_difference = Float::with_val(
                cfg.precision_bits,
                &backfilled.eigenvalue - &refreshed.eigenvalue,
            )
            .abs();
            assert!(
                refresh_difference < replay_tolerance,
                "refresh changed the canonical eigenvalue by {refresh_difference:?}"
            );
            for (index, (fresh, replayed)) in backfilled
                .distances
                .iter()
                .zip(&reused.distances)
                .enumerate()
            {
                replay_fields.push((format!("distance[{index}]"), &fresh.value, &replayed.value));
            }
            for (field, fresh, replayed) in replay_fields {
                let replay_difference = Float::with_val(cfg.precision_bits, fresh - replayed).abs();
                assert!(
                    replay_difference < replay_tolerance,
                    "stored-decimal {field} replay drifted by {replay_difference:?}"
                );
            }
            std::fs::remove_dir_all(root).unwrap();
        }

        /// The HP and f64 crossing detectors agree on the same profile.
        ///
        /// The two tiers must not disagree about whether the distance
        /// integrand is smooth: that verdict decides which integration rule
        /// is trustworthy for a configuration, and the campaign runs at HP
        /// while exploratory work runs at f64.
        #[test]
        fn hp_and_f64_crossing_detection_agree() {
            let prec = 192;
            let lambda_hp = Float::with_val(prec, 4u32);

            // tau <= 1 on [1, lambda]: f == 1 never crosses it.
            let smooth = hp::target_crossings(
                |_u: &Float| Float::with_val(prec, 1u32),
                &lambda_hp,
                200,
                GridVariable::U,
                prec,
            )
            .unwrap();
            let smooth_f64 =
                super::target_crossings_f64(|_| 1.0, 4.0, 200, GridVariable::U).unwrap();
            assert_eq!(smooth.crossings(), smooth_f64.crossings());
            assert!(smooth.integrand_appears_smooth());
            assert_eq!(smooth.initial_sign, smooth_f64.initial_sign);

            // A deliberate crossing must be seen by both tiers.
            let crossing = hp::target_crossings(
                |u: &Float| {
                    let mut value = Float::with_val(prec, 2u32);
                    value -= u;
                    value * Float::with_val(prec, 0.5)
                },
                &lambda_hp,
                200,
                GridVariable::U,
                prec,
            )
            .unwrap();
            let crossing_f64 =
                super::target_crossings_f64(|u| 0.5 * (2.0 - u), 4.0, 200, GridVariable::U)
                    .unwrap();
            assert_eq!(crossing.crossings(), crossing_f64.crossings());
            assert!(!crossing.integrand_appears_smooth());
            assert!(hp::target_crossings(
                |_: &Float| Float::with_val(prec, 1u32),
                &lambda_hp,
                1,
                GridVariable::U,
                prec
            )
            .is_err());
        }

        /// HP and f64 paths agree on the same distance at matching inputs.
        #[test]
        fn hp_and_f64_distances_agree() {
            let prec = 192;
            let lambda_hp = Float::with_val(prec, 9u32);
            let f_hp = hp_constant_one(&lambda_hp, 3, prec);
            let alpha = Float::with_val(prec, 0.5);
            let hp_value = hp::distance_to_target(
                |u: &Float| f_hp.eval(u),
                &lambda_hp,
                &alpha,
                WeightedIntegrationRule::UniformGrid {
                    scheme: UniformGridScheme::Midpoint,
                    variable: GridVariable::LogU,
                    steps: 5_000,
                },
                prec,
            )
            .unwrap()
            .value
            .to_f64();

            let f = super::constant_one_eigenfunction(9.0, 3);
            let f64_value = crate::distance::distance_to_target_f64(
                |u| f.eval(u),
                9.0,
                0.5,
                WeightedIntegrationRule::UniformGrid {
                    scheme: UniformGridScheme::Midpoint,
                    variable: GridVariable::LogU,
                    steps: 5_000,
                },
            )
            .unwrap()
            .value;
            assert!(
                (hp_value - f64_value).abs() < 1e-12,
                "hp {hp_value} vs f64 {f64_value}"
            );
        }
    }
}
