//! Private-only diagnostics comparing the retained even CCM state with the
//! runtime target. The artifact kind never reaches a public surface.
use super::*;
use rug::float::Round;

pub const TARGET_COMPARISON_KIND: &str = "ccm_target_comparison_analysis";
pub const TARGET_COMPARISON_SEMANTICS: &str = "ccm-runtime-target-comparison-v0.16.0-v1";
const TARGET_COMPARISON_SCOPE: &str = "trapezoid sums on a symmetric log grid at two resolutions; the projection quantities use the exact stored even-sector matrix restricted to the retained modes; empirical grid agreement, not integral error bounds";
const CRITICAL_LINE_OVERSAMPLING: usize = 8;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PortableTargetProjection {
    /// `|P target|^2 / |target|^2` for the projection onto the retained modes.
    pub captured_energy_fraction: String,
    /// `|<P target, f>| / (|P target| |f|)`.
    pub projection_overlap: String,
    /// `1 - projection_overlap^2`.
    pub overlap_defect: String,
    /// Weil form of the projection divided by its squared norm.
    pub rayleigh_quotient: String,
    /// Upper value of `(R - mu_1) / (mu_2 - mu_1)` from the eigenvalue
    /// enclosures, when `mu_2 > mu_1` is resolved.
    pub normalized_excess_upper: Option<String>,
    /// Whether `overlap_defect <= normalized_excess_upper` holds.
    pub spectral_bound_holds: Option<bool>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PortableTargetComparisonLevel {
    /// `J`: nodes per half grid beyond `t = 0`.
    pub half_points: usize,
    /// Grid step `delta` in `t`.
    pub step: String,
    /// `int_0^{log lambda} |g(e^t)| e^{t/2} dt`, i.e. `int_1^lambda |g| u^{-1/2} du`.
    pub weighted_l1_upper: String,
    /// `int_0^{log lambda} |g(e^-t)| e^{t/2} dt`, the same norm of `g(1/u)`.
    pub weighted_l1_lower: String,
    pub weighted_l1_sum: String,
    /// Part of `weighted_l1_upper` from `u` in `[lambda/2, lambda]`, split at
    /// `log(lambda/2)` with the integrand linearly interpolated; absent when
    /// `lambda <= 2`.
    pub edge_contribution: Option<String>,
    /// `max_j |g(e^{t_j})|` over the whole grid.
    pub maximum_absolute_residual: String,
    /// `max_{t_j >= 0} |target(e^t) - target(e^-t)|`.
    pub target_symmetry_defect: String,
    /// `max_{t_j >= 0} |f(e^t) - f(e^-t)|`; zero by parity up to rounding.
    pub state_symmetry_defect: String,
    /// `L^2(d*u)` norms on `[1/lambda, lambda]`.
    pub target_norm: String,
    pub state_norm: String,
    /// `|<target, f>| / (|target| |f|)` in `L^2(d*u)`.
    pub overlap: String,
    pub projection: Option<PortableTargetProjection>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PortableCriticalLineMaximum {
    /// `Re s` of the line.
    pub real_part: String,
    /// Sampled maximum of `|int g(u) u^s d*u|` over `Im s`.
    pub maximum: String,
    /// `|Im s|` at the sampled maximum.
    pub frequency: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PortableTargetComparison {
    pub schema_version: u32,
    pub semantics: String,
    pub claim_scope: String,
    /// SHA-256 of the canonical private runtime target specification.
    pub target_definition_digest: String,
    pub lambda_squared: String,
    pub n_modes: usize,
    pub precision_bits: u32,
    pub eigenpair_content_digest: String,
    pub even_matrix_content_digest: String,
    /// Unnormalized target at `u = 1`.
    pub target_normalization: String,
    pub grid: String,
    /// Coarse level `J` and its refinement `2J`.
    pub levels: Vec<PortableTargetComparisonLevel>,
    /// Both critical lines on the finer level.
    pub critical_lines: Vec<PortableCriticalLineMaximum>,
    pub critical_line_method: String,
    /// Directed enclosures of the two lowest even-sector eigenvalues.
    pub lowest_eigenvalues: Vec<crate::ccm::hp::CcmCheckpointEigenvalue>,
    /// Residual of the grid projection of `f` as an eigenvector of the stored
    /// even-sector matrix, relative to the matrix scale.
    pub basis_consistency_residual: Option<String>,
    pub projection_limitation: Option<String>,
    pub outcome: String,
}

/// Grid size for a configuration: the coarse level resolves every retained
/// mode exactly for the state (a trigonometric polynomial of degree `N`).
pub fn default_half_points(n_modes: usize) -> usize {
    256usize.max(n_modes + 1)
}

fn text(value: &Float, prec: u32) -> String {
    decimal(&Float::with_val(prec, value), prec)
}

/// In-place iterative radix-2 FFT (decimation in time).
fn fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut length = 2;
    while length <= n {
        let angle = -2.0 * std::f64::consts::PI / length as f64;
        for start in (0..n).step_by(length) {
            for k in 0..length / 2 {
                let (s, c) = (angle * k as f64).sin_cos();
                let (a, b) = (start + k, start + k + length / 2);
                let tr = re[b] * c - im[b] * s;
                let ti = re[b] * s + im[b] * c;
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
            }
        }
        length <<= 1;
    }
}

/// Sampled maximum of `|sum_j w_j g_j e^{sigma t_j} e^{i omega t_j}|` over
/// `omega`, by a zero-padded DFT. Returns `(maximum, |omega|)`.
fn critical_line_maximum(weighted: &[f64], step: f64) -> (f64, f64) {
    let size = (weighted.len() * CRITICAL_LINE_OVERSAMPLING).next_power_of_two();
    let scale = weighted.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    if scale == 0.0 {
        return (0.0, 0.0);
    }
    let mut re = vec![0.0; size];
    let mut im = vec![0.0; size];
    for (slot, value) in re.iter_mut().zip(weighted) {
        *slot = value / scale;
    }
    fft(&mut re, &mut im);
    let (mut best, mut at) = (0.0f64, 0usize);
    for m in 0..=size / 2 {
        let magnitude = re[m].hypot(im[m]);
        if magnitude > best {
            best = magnitude;
            at = m;
        }
    }
    (
        best * scale,
        2.0 * std::f64::consts::PI * at as f64 / (size as f64 * step),
    )
}

/// Levels, critical lines, basis-consistency residual and any projection limitation.
type ComparisonParts = (
    Vec<PortableTargetComparisonLevel>,
    Vec<PortableCriticalLineMaximum>,
    Option<String>,
    Option<String>,
);

struct Samples {
    /// `t_j` for `j = -J..J` at the finer level, index `j + J`.
    t: Vec<Float>,
    target: Vec<Float>,
    state: Vec<Float>,
}

/// Compute the comparison. `coefficients` are the normalized even
/// coefficients of `f` (`j = 0..N`), `even_matrix` the stored `(N+1)^2`
/// even-sector matrix, `lowest` the enclosures of its two lowest eigenvalues.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compare_with_target(
    target: &dyn Fn(&Float) -> Result<Float>,
    lambda: &Float,
    coefficients: &[Float],
    even_matrix: Option<&[Float]>,
    lowest: &[crate::ccm::hp::CcmCheckpointEigenvalue],
    half_points: usize,
    prec: u32,
) -> Result<ComparisonParts> {
    let n = coefficients.len();
    anyhow::ensure!(
        n >= 1 && half_points >= 2,
        "comparison needs modes and a grid"
    );
    if let Some(matrix) = even_matrix {
        anyhow::ensure!(matrix.len() == n * n, "even-sector matrix shape mismatch");
    }
    let working = prec.saturating_add(GUARD_BITS);
    let fine = 2 * half_points;
    let length = log_length(lambda, working);
    let half = Float::with_val(working, &length / 2u32);
    let step = Float::with_val(working, &half / fine as u32);
    let two_pi = Float::with_val(working, Constant::Pi) * 2u32;
    // Even basis on [-L/2, L/2]: e_0 = L^{-1/2}, e_k = (2/L)^{1/2} (-1)^k cos(k phi).
    let inv_sqrt_length = Float::with_val(working, length.clone().recip_sqrt());
    let sqrt_two_over_length = Float::with_val(
        working,
        &inv_sqrt_length * Float::with_val(working, 2u32).sqrt(),
    );
    let basis_scale = |k: usize| -> Float {
        let mut s = if k == 0 {
            inv_sqrt_length.clone()
        } else {
            sqrt_two_over_length.clone()
        };
        if k % 2 == 1 {
            s = -s;
        }
        s
    };
    // Levels: index 0 is the coarse grid (every other node), 1 the fine grid.
    let strides = [2usize, 1usize];
    let mut projected_target = vec![vec![Float::with_val(working, 0); n]; 2];
    let mut projected_state = vec![vec![Float::with_val(working, 0); n]; 2];
    let mut samples = Samples {
        t: Vec::with_capacity(2 * fine + 1),
        target: Vec::with_capacity(2 * fine + 1),
        state: Vec::with_capacity(2 * fine + 1),
    };
    let mut cosines = vec![Float::with_val(working, 0); n];
    for j in -(fine as i64)..=(fine as i64) {
        let t = Float::with_val(working, &step * j);
        let u = Float::with_val(working, t.clone().exp());
        let target_value = Float::with_val(working, target(&u)?);
        let mut phi = Float::with_val(working, &two_pi * &t);
        phi /= &length;
        cosines[0] = Float::with_val(working, 1u32);
        if n > 1 {
            cosines[1] = Float::with_val(working, phi.cos());
        }
        for k in 2..n {
            let mut next = Float::with_val(working, &cosines[1] * &cosines[k - 1]);
            next *= 2u32;
            next -= &cosines[k - 2];
            cosines[k] = next;
        }
        // f(t) = c_0 + 2 sum_k c_k (-1)^k cos(k phi), normalized so f(1) = 1.
        let mut state = Float::with_val(working, &coefficients[0]);
        for k in 1..n {
            let mut term = Float::with_val(working, &coefficients[k] * &cosines[k]);
            term *= 2u32;
            if k % 2 == 1 {
                state -= term;
            } else {
                state += term;
            }
        }
        let index = (j + fine as i64) as usize;
        for (level, stride) in strides.iter().enumerate() {
            if !index.is_multiple_of(*stride) {
                continue;
            }
            let mut weight = Float::with_val(working, &step * *stride as u32);
            if j.unsigned_abs() as usize == fine {
                weight /= 2u32;
            }
            for k in 0..n {
                let basis = Float::with_val(working, &basis_scale(k) * &cosines[k]);
                let mut a = Float::with_val(working, &basis * &target_value);
                a *= &weight;
                projected_target[level][k] += a;
                let mut b = Float::with_val(working, &basis * &state);
                b *= &weight;
                projected_state[level][k] += b;
            }
        }
        samples.t.push(t);
        samples.target.push(target_value);
        samples.state.push(state);
    }
    let center = fine;
    anyhow::ensure!(
        Float::with_val(working, &samples.state[center] - 1u32).abs()
            < Float::with_val(working, 1u32) >> (prec / 2).max(32),
        "reconstructed state does not satisfy f(1) = 1"
    );
    let mut levels = Vec::new();
    let mut consistency = None;
    let mut limitation = None;
    for (level, stride) in strides.iter().enumerate() {
        let h = Float::with_val(working, &step * *stride as u32);
        let count = fine / stride;
        let at = |j: i64| center as i64 + j * *stride as i64;
        let residual = |i: i64| -> Float {
            let i = i as usize;
            Float::with_val(working, &samples.state[i] - &samples.target[i])
        };
        let mut upper = Float::with_val(working, 0);
        let mut lower = Float::with_val(working, 0);
        let mut target_defect = Float::with_val(working, 0);
        let mut state_defect = Float::with_val(working, 0);
        let mut maximum = Float::with_val(working, 0);
        let weight_at = |j: usize| -> Float {
            let mut w = h.clone();
            if j == 0 || j == count {
                w /= 2u32;
            }
            w
        };
        for j in 0..=count {
            let e = Float::with_val(working, &samples.t[at(j as i64) as usize] / 2u32).exp();
            let plus = residual(at(j as i64)).abs();
            let minus = residual(at(-(j as i64))).abs();
            maximum = maximum.max(&plus).max(&minus);
            let w = weight_at(j);
            upper += Float::with_val(working, &plus * &e) * &w;
            lower += Float::with_val(working, &minus * &e) * &w;
            let (p, m) = (at(j as i64) as usize, at(-(j as i64)) as usize);
            target_defect = target_defect
                .max(&Float::with_val(working, &samples.target[p] - &samples.target[m]).abs());
            state_defect = state_defect
                .max(&Float::with_val(working, &samples.state[p] - &samples.state[m]).abs());
        }
        // Edge contribution over u in [lambda/2, lambda].
        let edge = if lambda > &2u32 {
            let split = Float::with_val(working, &half - Float::with_val(working, 2u32).ln());
            let integrand = |j: usize| -> Float {
                let i = at(j as i64) as usize;
                let e = Float::with_val(working, &samples.t[i] / 2u32).exp();
                Float::with_val(working, residual(i as i64).abs() * e)
            };
            let first = (0..=count)
                .find(|&j| samples.t[at(j as i64) as usize] >= split)
                .unwrap_or(count);
            let mut total = Float::with_val(working, 0);
            for j in first..count {
                let mut piece = Float::with_val(working, integrand(j) + integrand(j + 1));
                piece *= &h;
                piece /= 2u32;
                total += piece;
            }
            if first > 0 {
                let (t0, t1) = (
                    &samples.t[at(first as i64 - 1) as usize],
                    &samples.t[at(first as i64) as usize],
                );
                let numerator = Float::with_val(working, &split - t0);
                let denominator = Float::with_val(working, t1 - t0);
                let fraction = Float::with_val(working, &numerator / &denominator);
                let left = integrand(first - 1);
                let right = integrand(first);
                let mut mid = Float::with_val(working, &right - &left);
                mid *= &fraction;
                mid += &left;
                let mut piece = Float::with_val(working, &mid + &right);
                piece *= Float::with_val(working, t1 - &split);
                piece /= 2u32;
                total += piece;
            }
            Some(text(&total, prec))
        } else {
            None
        };
        // L^2(dt) norms and inner product on the full grid.
        let mut target_square = Float::with_val(working, 0);
        let mut state_square = Float::with_val(working, 0);
        let mut inner = Float::with_val(working, 0);
        for j in -(count as i64)..=(count as i64) {
            let i = at(j) as usize;
            // Full-grid trapezoid: only the two endpoints are halved.
            let mut w = h.clone();
            if j.unsigned_abs() as usize == count {
                w /= 2u32;
            }
            target_square += Float::with_val(working, samples.target[i].clone().square()) * &w;
            state_square += Float::with_val(working, samples.state[i].clone().square()) * &w;
            inner += Float::with_val(working, &samples.target[i] * &samples.state[i]) * &w;
        }
        let target_norm = Float::with_val(working, target_square.clone().sqrt());
        let state_norm = Float::with_val(working, state_square.sqrt());
        let overlap = Float::with_val(
            working,
            inner.abs() / Float::with_val(working, &target_norm * &state_norm),
        );
        // Projection onto the retained modes and the Weil form of it.
        let projection = match even_matrix {
            None => None,
            Some(matrix) => {
                let a = &projected_target[level];
                let b = &projected_state[level];
                let dot = |x: &[Float], y: &[Float]| {
                    x.iter()
                        .zip(y)
                        .fold(Float::with_val(working, 0), |s, (p, q)| {
                            s + Float::with_val(working, p * q)
                        })
                };
                let apply = |x: &[Float]| -> Vec<Float> {
                    (0..n)
                        .map(|r| {
                            matrix[r * n..(r + 1) * n]
                                .iter()
                                .zip(x)
                                .fold(Float::with_val(working, 0), |s, (m, v)| {
                                    s + Float::with_val(working, m * v)
                                })
                        })
                        .collect()
                };
                let aa = dot(a, a);
                let bb = dot(b, b);
                // The grid projection of f is exact for its N modes, so it must be
                // an eigenvector of the stored matrix; this guards the basis convention.
                let eb = apply(b);
                let rayleigh_state = Float::with_val(working, dot(b, &eb) / &bb);
                let scale = matrix.iter().fold(Float::with_val(working, 0), |m, v| {
                    m.max(&Float::with_val(working, v.abs_ref()))
                });
                let mut residual_square = Float::with_val(working, 0);
                for (eb_k, b_k) in eb.iter().zip(b) {
                    let r = Float::with_val(
                        working,
                        eb_k - Float::with_val(working, &rayleigh_state * b_k),
                    );
                    residual_square += r.square();
                }
                let relative = Float::with_val(
                    working,
                    residual_square.sqrt() / Float::with_val(working, &scale * bb.clone().sqrt()),
                );
                if level == strides.len() - 1 {
                    consistency = Some(text(&relative, prec));
                }
                if relative > (Float::with_val(working, 1u32) >> 40u32) {
                    limitation = Some(
                        "the state's grid projection is not an eigenvector of the stored even-sector matrix; projection quantities withheld".to_owned(),
                    );
                    None
                } else {
                    let ea = apply(a);
                    let rayleigh = Float::with_val(working, dot(a, &ea) / &aa);
                    let projection_overlap = Float::with_val(
                        working,
                        dot(a, b).abs()
                            / Float::with_val(working, Float::with_val(working, &aa * &bb).sqrt()),
                    );
                    let defect = Float::with_val(
                        working,
                        1u32 - Float::with_val(working, projection_overlap.clone().square()),
                    );
                    let parse = |s: &str, round: Round| -> Option<Float> {
                        Some(Float::with_val_round(working, Float::parse(s).ok()?, round).0)
                    };
                    let bounds = (|| -> Option<(Float, Float, Float)> {
                        let first = lowest.first()?;
                        let second = lowest.get(1)?;
                        Some((
                            parse(&first.lower, Round::Down)?,
                            parse(&first.upper, Round::Up)?,
                            parse(&second.lower, Round::Down)?,
                        ))
                    })();
                    // Rounding allowance of the computed quotient: n * max|E| ulps.
                    let rounding = Float::with_val(working, &scale * n as u32) >> (working - 8);
                    let excess = bounds.and_then(|(mu1_lower, mu1_upper, mu2_lower)| {
                        let gap =
                            Float::with_val_round(working, &mu2_lower - &mu1_upper, Round::Down).0;
                        (gap > 0).then(|| {
                            let numerator =
                                Float::with_val_round(working, &rayleigh - &mu1_lower, Round::Up).0;
                            let allowance =
                                Float::with_val_round(working, &rounding / &gap, Round::Up).0;
                            (
                                Float::with_val_round(working, &numerator / &gap, Round::Up).0,
                                allowance,
                            )
                        })
                    });
                    let defect_allowance = Float::with_val(working, 1u32) >> (working - 8);
                    Some(PortableTargetProjection {
                        captured_energy_fraction: text(
                            &Float::with_val(working, &aa / &target_square),
                            prec,
                        ),
                        projection_overlap: text(&projection_overlap, prec),
                        overlap_defect: text(&defect, prec),
                        rayleigh_quotient: text(&rayleigh, prec),
                        spectral_bound_holds: excess.as_ref().map(|(bound, allowance)| {
                            Float::with_val(working, &defect - &defect_allowance)
                                <= Float::with_val(working, bound + allowance)
                        }),
                        normalized_excess_upper: excess
                            .as_ref()
                            .map(|(bound, _)| text(bound, prec)),
                    })
                }
            }
        };
        levels.push(PortableTargetComparisonLevel {
            half_points: count,
            step: text(&h, prec),
            weighted_l1_sum: text(&Float::with_val(working, &upper + &lower), prec),
            weighted_l1_upper: text(&upper, prec),
            weighted_l1_lower: text(&lower, prec),
            edge_contribution: edge,
            maximum_absolute_residual: text(&maximum, prec),
            target_symmetry_defect: text(&target_defect, prec),
            state_symmetry_defect: text(&state_defect, prec),
            target_norm: text(&target_norm, prec),
            state_norm: text(&state_norm, prec),
            overlap: text(&overlap, prec),
            projection,
        });
    }
    // Critical lines Re s = +1/2 and -1/2 on the fine grid.
    let step_f64 = step.to_f64();
    let mut critical_lines = Vec::new();
    for (sigma, label) in [(0.5f64, "1/2"), (-0.5f64, "-1/2")] {
        let weighted = (0..samples.t.len())
            .map(|i| {
                let t = samples.t[i].to_f64();
                let g = Float::with_val(working, &samples.state[i] - &samples.target[i]).to_f64();
                let w = if i == 0 || i + 1 == samples.t.len() {
                    step_f64 / 2.0
                } else {
                    step_f64
                };
                g * (sigma * t).exp() * w
            })
            .collect::<Vec<_>>();
        let (maximum, frequency) = critical_line_maximum(&weighted, step_f64);
        critical_lines.push(PortableCriticalLineMaximum {
            real_part: label.to_owned(),
            maximum: format!("{maximum:.17e}"),
            frequency: format!("{frequency:.17e}"),
        });
    }
    Ok((levels, critical_lines, consistency, limitation))
}

/// Capture the comparison as a private `ccm_target_comparison_analysis`
/// artifact bound to the canonical even eigenpair and the stored even-sector
/// matrix. The runtime target specification enters only through its digest.
pub fn capture_target_comparison_via_cache(
    params: &crate::ccm::CcmParams,
    cfg: &crate::ccm::hp::HighPrecConfig,
    even_matrix: &crate::ccm::prefix::RetainedEvenMatrix,
    cache: &xc_cache::ArtifactCacheContext<'_>,
) -> Result<xc_cache::ArtifactExecutionCacheResult<PortableTargetComparison>> {
    use std::collections::BTreeMap;
    use xc_cache::{
        resolve_or_compute_json_artifact_with_dependencies, ArtifactExecutionCacheRequest,
        CacheQuality, DependencyRef, SemanticKeyEnvelope, ToolkitVersion,
    };
    let prec = cfg.precision_bits;
    let working = prec.saturating_add(GUARD_BITS);
    let target_spec = crate::target::TargetProfileSpec::from_environment()?;
    let target_definition_digest = target_spec.digest()?;
    let lambda_sq_identity = lambda_squared_identity(params);
    let lambda_sq = crate::ccm::hp::lambda_squared_value_hp(params, working)?;
    anyhow::ensure!(lambda_sq > 1u32, "target comparison needs lambda^2 > 1");
    let lambda = lambda_sq.sqrt();
    anyhow::ensure!(
        even_matrix.dimension() == params.n_modes + 1,
        "even-sector matrix does not match the configuration"
    );
    let canonical =
        crate::ccm::hp::resolve_canonical_even_eigenstate_via_cache(params, cfg, cache)?;
    let half_points = default_half_points(params.n_modes);
    // The two lowest even eigenvalues come from the same cached checkpoint
    // spectra Ultra records (identical arguments), so they are computed once.
    let dimension = even_matrix.dimension();
    let spectra = crate::ccm::hp::checkpoint_low_spectra_via_cache(
        even_matrix,
        &crate::ccm::capture::checkpoint_spectrum_ladder(dimension),
        3,
        cache,
    )?;
    let spectra_manifest = spectra
        .produced_manifest
        .clone()
        .or_else(|| spectra.reused_manifest.clone());
    let lowest: Vec<crate::ccm::hp::CcmCheckpointEigenvalue> = spectra
        .value
        .rows
        .iter()
        .find(|row| row.dimension == dimension)
        .map(|row| row.eigenvalues.iter().take(2).cloned().collect())
        .unwrap_or_default();
    let dependency = |manifest: &xc_cache::ArtifactManifest| DependencyRef {
        key: manifest.key.clone(),
        content_digest: manifest.content_digest.clone(),
        required_quality: CacheQuality::Validated,
    };
    let semantic_key = SemanticKeyEnvelope {
        schema_version: 1,
        artifact_kind: TARGET_COMPARISON_KIND.to_owned(),
        mathematical_semantics_version: TARGET_COMPARISON_SEMANTICS.to_owned(),
        resolved_mathematical_parameters: serde_json::json!({
            "target_definition_digest": target_definition_digest,
            "lambda_squared": lambda_sq_identity,
            "n_modes": params.n_modes,
            "precision_bits": prec,
            "eigenpair_content_digest": canonical.manifest.content_digest.0,
            "even_matrix_content_digest": even_matrix.manifest().content_digest.0,
            "half_points": half_points,
            "critical_line_oversampling": CRITICAL_LINE_OVERSAMPLING,
        }),
        normalization: Some("f(1)=1".to_owned()),
        target: Some(TARGET_COMPARISON_KIND.to_owned()),
        subspace: Some("even".to_owned()),
        source_data_identities: BTreeMap::new(),
        algorithm_semantics: Some("symmetric_log_grid_trapezoid_two_levels_v1".to_owned()),
    };
    let logical_key = format!(
        "ccm/target-comparison/{}/{}/{}/{}",
        lambda_sq_identity, params.n_modes, prec, half_points
    );
    let request = ArtifactExecutionCacheRequest {
        operation: "ccm.target_comparison.resolve_or_compute",
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
        minimum_reader_version: ToolkitVersion::parse(xc_cache::CLEAN_SLATE)?,
        maximum_reader_version: None,
        tags: BTreeMap::from([
            ("domain".to_owned(), "ccm".to_owned()),
            (
                "artifact".to_owned(),
                "target_comparison_analysis".to_owned(),
            ),
        ]),
        provenance_digest: None,
        production_sink: cache.production_sink,
    };
    let mut dependencies = vec![
        dependency(&canonical.manifest),
        dependency(even_matrix.manifest()),
    ];
    if let Some(manifest) = &spectra_manifest {
        dependencies.push(dependency(manifest));
    }
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
    Ok(resolve_or_compute_json_artifact_with_dependencies(
        &request,
        || {
            let target = crate::target::hp::TargetEvaluator::from_spec(&target_spec, working)
                .map_err(|e| xc_cache::CacheError::InvalidManifest(format!("{e:#}")))?;
            let computed = (|| -> Result<PortableTargetComparison> {
                target.validate_lambda(&lambda)?;
                let eigenfunction = WeilEigenfunction::from_v_basis(
                    &canonical.eigenvector,
                    params.n_modes,
                    &lambda,
                    prec,
                )?;
                let lowest = lowest.clone();
                let (levels, critical_lines, consistency, limitation) = compare_with_target(
                    &|u: &Float| target.try_value(u),
                    &lambda,
                    &eigenfunction.normalized_coefficients(),
                    Some(even_matrix.entries()),
                    &lowest,
                    half_points,
                    prec,
                )?;
                Ok(PortableTargetComparison {
                    schema_version: 1,
                    semantics: TARGET_COMPARISON_SEMANTICS.to_owned(),
                    claim_scope: TARGET_COMPARISON_SCOPE.to_owned(),
                    target_definition_digest: target_definition_digest.clone(),
                    lambda_squared: lambda_sq_identity.clone(),
                    n_modes: params.n_modes,
                    precision_bits: prec,
                    eigenpair_content_digest: canonical.manifest.content_digest.0.clone(),
                    even_matrix_content_digest: even_matrix.manifest().content_digest.0.clone(),
                    target_normalization: text(&target.normalization(), prec),
                    grid: "t = log u; t_j = j*delta, j = -J..J, J*delta = log(lambda); trapezoid with halved endpoints; coarse level J, fine level 2J".to_owned(),
                    levels,
                    critical_lines,
                    critical_line_method: format!(
                        "binary64 zero-padded DFT of trapezoid-weighted g(e^t) e^(sigma t) on the fine level, oversampling {CRITICAL_LINE_OVERSAMPLING}; sampled maximum over Im s"
                    ),
                    outcome: if limitation.is_some() {
                        "computed_with_limitation".to_owned()
                    } else {
                        "computed".to_owned()
                    },
                    lowest_eigenvalues: lowest,
                    basis_consistency_residual: consistency,
                    projection_limitation: limitation,
                })
            })();
            let value =
                computed.map_err(|e| xc_cache::CacheError::InvalidManifest(format!("{e:#}")))?;
            Ok((value, dependencies.clone()))
        },
        |value: &PortableTargetComparison| {
            if value.semantics != TARGET_COMPARISON_SEMANTICS
                || value.levels.len() != 2
                || value.critical_lines.len() != 2
            {
                return Err(xc_cache::CacheError::InvalidManifest(
                    "invalid target comparison payload".to_owned(),
                ));
            }
            Ok(())
        },
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: u32 = 256;

    /// Normalized even coefficients with f(1) = 1, the matching normalized
    /// sector vector x ~ (c_0, sqrt(2) c_k), and a two-level matrix having x as
    /// its mu_1 eigenvector and every orthogonal direction at mu_2.
    fn fixture(
        mu1: &str,
        mu2: &str,
    ) -> (
        Float,
        Vec<Float>,
        Vec<Float>,
        Vec<crate::ccm::hp::CcmCheckpointEigenvalue>,
    ) {
        let lambda = Float::with_val(P, 13u32).sqrt();
        let raw = [0.9f64, -0.21, 0.05, -0.013, 0.002];
        // f(1) = c_0 + 2 sum (-1)^k c_k.
        let mut at_one = Float::with_val(P, raw[0]);
        for (k, v) in raw.iter().enumerate().skip(1) {
            let term = Float::with_val(P, *v) * 2u32;
            if k % 2 == 1 {
                at_one -= term;
            } else {
                at_one += term;
            }
        }
        let coefficients: Vec<Float> = raw
            .iter()
            .map(|v| Float::with_val(P, *v) / &at_one)
            .collect();
        let n = coefficients.len();
        let mut x: Vec<Float> = coefficients
            .iter()
            .enumerate()
            .map(|(k, c)| {
                if k == 0 {
                    c.clone()
                } else {
                    Float::with_val(P, c * Float::with_val(P, 2u32).sqrt())
                }
            })
            .collect();
        let norm = x
            .iter()
            .fold(Float::with_val(P, 0), |s, v| s + Float::with_val(P, v * v))
            .sqrt();
        for v in &mut x {
            *v /= &norm;
        }
        let (m1, m2) = (
            Float::with_val(P, Float::parse(mu1).unwrap()),
            Float::with_val(P, Float::parse(mu2).unwrap()),
        );
        let mut matrix = vec![Float::with_val(P, 0); n * n];
        for r in 0..n {
            for c in 0..n {
                let mut value = Float::with_val(P, &x[r] * &x[c]) * Float::with_val(P, &m1 - &m2);
                if r == c {
                    value += &m2;
                }
                matrix[r * n + c] = value;
            }
        }
        // Directed enclosures that bracket the matrix eigenvalues, as the
        // stored-matrix eigensolver supplies them.
        let bound = |value: &Float| {
            let lower = Float::with_val(P, value - Float::with_val(P, value >> 200u32));
            let upper = Float::with_val(P, value + Float::with_val(P, value >> 200u32));
            crate::ccm::hp::CcmCheckpointEigenvalue {
                index: 0,
                lower: lower.to_string_radix_round(10, None, rug::float::Round::Down),
                upper: upper.to_string_radix_round(10, None, rug::float::Round::Up),
                resolution: crate::ccm::hp::StoredEigenvalueResolution::Resolved,
            }
        };
        let lowest = vec![bound(&m1), bound(&m2)];
        (lambda, coefficients, matrix, lowest)
    }

    fn state(coefficients: &[Float], lambda: &Float, u: &Float) -> Float {
        let length = log_length(lambda, P + GUARD_BITS);
        let phi = Float::with_val(
            P + GUARD_BITS,
            Float::with_val(P + GUARD_BITS, u.clone().ln())
                * Float::with_val(P + GUARD_BITS, rug::float::Constant::Pi)
                * 2u32,
        ) / length;
        let mut value = Float::with_val(P + GUARD_BITS, &coefficients[0]);
        for (k, c) in coefficients.iter().enumerate().skip(1) {
            let term = Float::with_val(
                P + GUARD_BITS,
                Float::with_val(P + GUARD_BITS, &phi * k as u32).cos() * c,
            ) * 2u32;
            if k % 2 == 1 {
                value -= term;
            } else {
                value += term;
            }
        }
        value
    }

    #[test]
    fn identical_target_has_zero_residual_and_the_ground_rayleigh_quotient() {
        let (lambda, coefficients, matrix, lowest) = fixture("1e-30", "1e-20");
        let target = |u: &Float| Ok(state(&coefficients, &lambda, u));
        let (levels, lines, consistency, limitation) = compare_with_target(
            &target,
            &lambda,
            &coefficients,
            Some(&matrix),
            &lowest,
            16,
            P,
        )
        .unwrap();
        assert!(limitation.is_none());
        assert!(Float::with_val(P, Float::parse(consistency.unwrap()).unwrap()) < 1e-60);
        for level in &levels {
            assert!(
                Float::with_val(P, Float::parse(&level.weighted_l1_sum).unwrap()).abs() < 1e-60
            );
            assert!(
                (Float::with_val(P, Float::parse(&level.overlap).unwrap()) - 1u32).abs() < 1e-60
            );
            let projection = level.projection.as_ref().unwrap();
            let rayleigh = Float::with_val(P, Float::parse(&projection.rayleigh_quotient).unwrap());
            assert!((rayleigh - Float::with_val(P, Float::parse("1e-30").unwrap())).abs() < 1e-60);
            assert_eq!(projection.spectral_bound_holds, Some(true));
        }
        assert!(lines
            .iter()
            .all(|line| line.maximum.parse::<f64>().unwrap() < 1e-60));
    }

    #[test]
    fn full_grid_norm_is_exact_for_the_retained_cosine_polynomial() {
        let (lambda, coefficients, matrix, lowest) = fixture("1e-30", "1e-20");
        let target = |u: &Float| Ok(state(&coefficients, &lambda, u));
        let (levels, _, _, _) = compare_with_target(
            &target,
            &lambda,
            &coefficients,
            Some(&matrix),
            &lowest,
            16,
            P,
        )
        .unwrap();
        // ||f||^2 = L (c_0^2 + 2 sum c_k^2) on [-L/2, L/2].
        let length = log_length(&lambda, P + GUARD_BITS);
        let mut expected = Float::with_val(P + GUARD_BITS, coefficients[0].clone().square());
        for c in &coefficients[1..] {
            expected += Float::with_val(P + GUARD_BITS, c.clone().square()) * 2u32;
        }
        expected *= &length;
        for level in &levels {
            let norm = Float::with_val(P, Float::parse(&level.state_norm).unwrap());
            let square = Float::with_val(P + GUARD_BITS, norm.square());
            assert!(
                Float::with_val(P, &square - &expected).abs()
                    < Float::with_val(P, &expected * 1e-60)
            );
            let fraction = Float::with_val(
                P,
                Float::parse(&level.projection.as_ref().unwrap().captured_energy_fraction).unwrap(),
            );
            assert!((fraction - 1u32).abs() < 1e-60);
        }
    }

    #[test]
    fn two_level_spectrum_attains_the_overlap_bound_and_symmetry_is_measured() {
        let (lambda, coefficients, matrix, lowest) = fixture("1e-30", "1e-20");
        // Perturb by an even mode (k = 1) and an asymmetric term in t.
        let target = |u: &Float| {
            let t = Float::with_val(P + 64, u.clone().ln());
            let length = log_length(&lambda, P + 64);
            let phi = Float::with_val(
                P + 64,
                &t * Float::with_val(P + 64, rug::float::Constant::Pi) * 2u32,
            ) / &length;
            Ok(state(&coefficients, &lambda, u) + Float::with_val(P + 64, phi.cos()) * 0.01)
        };
        let (levels, _, _, _) = compare_with_target(
            &target,
            &lambda,
            &coefficients,
            Some(&matrix),
            &lowest,
            16,
            P,
        )
        .unwrap();
        for level in &levels {
            let projection = level.projection.as_ref().unwrap();
            let defect = Float::with_val(P, Float::parse(&projection.overlap_defect).unwrap());
            let excess = Float::with_val(
                P,
                Float::parse(projection.normalized_excess_upper.as_ref().unwrap()).unwrap(),
            );
            assert!(defect > 1e-6);
            // Exact for a two-level spectrum: 1 - rho^2 = (R - mu1)/(mu2 - mu1).
            assert!(
                Float::with_val(P, &defect - &excess).abs() < Float::with_val(P, &defect * 1e-40)
            );
            // The perturbation is even in t, so both halves agree.
            assert_eq!(level.weighted_l1_upper, level.weighted_l1_lower);
            assert!(
                Float::with_val(P, Float::parse(&level.target_symmetry_defect).unwrap()) < 1e-60
            );
        }
    }

    #[test]
    fn fft_matches_direct_transform() {
        let samples = [0.3, -1.2, 0.7, 2.5, -0.4];
        let step = 0.25;
        let (maximum, frequency) = critical_line_maximum(&samples, step);
        let size = (samples.len() * CRITICAL_LINE_OVERSAMPLING).next_power_of_two();
        let mut best = 0.0f64;
        for m in 0..=size / 2 {
            let omega = 2.0 * std::f64::consts::PI * m as f64 / (size as f64 * step);
            let (mut re, mut im) = (0.0, 0.0);
            for (j, v) in samples.iter().enumerate() {
                re += v * (omega * j as f64 * step).cos();
                im -= v * (omega * j as f64 * step).sin();
            }
            best = best.max(re.hypot(im));
        }
        assert!((maximum - best).abs() < 1e-12 * best);
        assert!(frequency >= 0.0);
    }
}
