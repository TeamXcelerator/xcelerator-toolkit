// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! CCM "Zeta Spectral Triples" construction.
//!
//! Implements the CCM operator family `D_log(λ, N)`.
//! (27 Nov 2025). Numerical experiments compare its finite spectra with
//! Riemann zeta zeros. Convergence as `λ, N → ∞` remains an open obligation;
//! the construction and software do not prove that limit.
//!
//! ## Pipeline
//!
//! 1. Sieve prime powers `k ≤ λ²`.
//! 2. Build the Weil quadratic form matrix `τ_{n,m}` for `n,m ∈ {-N,…,N}`.
//! 3. Solve the reduced even sector and lift its lowest eigenvector `ξ`.
//!    This selects an even Ritz state; it does not prove full-space ground selection.
//! 4. The eigenvalues of `D_log(λ, N)` are the zeros of the rational
//!    function `R(z) = Σ ξ_j / (z − 2πj/L)`.

use anyhow::{anyhow, Result};
use std::time::Instant;

#[cfg(feature = "hp")]
pub mod hp;

#[cfg(feature = "hp")]
pub mod evidence;

#[cfg(feature = "hp")]
pub mod certified_roots;

#[cfg(feature = "hp")]
pub mod sector_gap_certificate;

#[cfg(feature = "arb")]
mod arb_bridge;

#[cfg(feature = "arb")]
pub mod cutoff_free;

#[cfg_attr(
    not(test),
    deprecated(
        note = "Legacy templates do not construct production CCM semantic keys; use managed CCM APIs for production artifact reuse."
    )
)]
pub mod artifacts;
pub mod convergence;

pub mod capture;
mod native_secular;
#[cfg(feature = "hp")]
pub mod prefix;
pub mod rank_one;
pub mod reproduction;
#[cfg(feature = "hp")]
pub mod research;
pub mod window;

/// How λ² is represented and processed.
///
/// Both `value_u64` and `value_f64` are always populated. The
/// `is_integer` flag controls which path the HP computation takes:
///
/// - `is_integer = true`: uses `value_u64` for exact HP promotion
///   (`Float::with_val(prec, value_u64)`) — full working precision,
///   no representation error. This is the correct path for the publication
///   configs (13, 100, 1000, etc.).
///
/// - `is_integer = false`: uses the decimal `format!("{:.17e}", value_f64)`
///   (18 significant digits) as the HP cutoff. This decimal differs from both
///   a caller's original decimal and the exact binary64 value. Its error is
///   amplified in relative terms by `ln(lambda^2)` near one; no fixed number of
///   accurate logarithm digits is promised. Use `hp::ExactLambdaSquaredHp`
///   and `hp::localized_weil_form_exact_hp` for an exact decimal cutoff.
///
/// Having both values always available allows optimizations: the u64
/// is always used for prime sieving (`prime_powers_up_to(value_u64)`),
/// the f64 is always used for display and f64-tier computation, and
/// the bool selects the HP promotion path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LambdaSq {
    /// Integer form: `⌊λ²⌋`. Used for prime cutoff, cache keys in
    /// integer mode, and exact HP promotion when `is_integer = true`.
    pub value_u64: u64,
    /// Float form: the full-precision f64 λ² value. Used for display,
    /// f64-tier computation, cache keys in fractional mode, and HP
    /// promotion when `is_integer = false`.
    pub value_f64: f64,
    /// `true` = integer mode (HP uses `value_u64`).
    /// `false` = fractional mode (HP uses `value_f64`).
    pub is_integer: bool,
}

impl LambdaSq {
    /// Construct in integer mode from a u64 value.
    pub fn integer(v: u64) -> Self {
        Self {
            value_u64: v,
            value_f64: v as f64,
            is_integer: true,
        }
    }

    /// Construct in fractional mode from an f64 value.
    pub fn fractional(v: f64) -> Self {
        Self {
            value_u64: v.floor() as u64,
            value_f64: v,
            is_integer: false,
        }
    }

    /// Construct in fractional mode but with an explicit u64 override
    /// (for testing: e.g. λ²=15.0 fractional with value_u64=15).
    pub fn fractional_with_int(value_f64: f64, value_u64: u64) -> Self {
        Self {
            value_u64,
            value_f64,
            is_integer: false,
        }
    }

    /// Mode string for JSON metadata: `"integer"` or `"fractional"`.
    pub fn mode_str(&self) -> &'static str {
        if self.is_integer {
            "integer"
        } else {
            "fractional"
        }
    }

    /// Canonical string for cache filenames.
    /// Integer: `"13"`, Fractional: `"12p5"` (dot replaced with `p`,
    /// trailing zeros stripped).
    pub fn filename_str(&self) -> String {
        if self.is_integer {
            format!("{}", self.value_u64)
        } else {
            let s = format!("{}", self.value_f64);
            if s.contains('.') {
                let trimmed = s.trim_end_matches('0');
                let trimmed = if trimmed.ends_with('.') {
                    format!("{}0", trimmed)
                } else {
                    trimmed.to_string()
                };
                trimmed.replace('.', "p")
            } else {
                format!("{}p0", s)
            }
        }
    }

    /// Parse a filename fragment back into a `LambdaSq`.
    /// `"13"` → integer(13), `"12p5"` → fractional(12.5).
    pub fn from_filename_str(s: &str) -> Option<Self> {
        let parsed = if s.contains('p') {
            let f_str = s.replace('p', ".");
            let v: f64 = f_str.parse().ok()?;
            if !v.is_finite() {
                return None;
            }
            LambdaSq::fractional(v)
        } else {
            LambdaSq::integer(s.parse().ok()?)
        };
        (parsed.filename_str() == s).then_some(parsed)
    }
}

/// Parameters for a single CCM run.
#[derive(Debug, Clone)]
pub struct CcmParams {
    /// `λ²` — either exact integer or fractional f64.
    pub lambda_sq: LambdaSq,

    /// Mode cutoff N. Matrix dimension is `2N+1`.
    pub n_modes: usize,
}

impl CcmParams {
    /// Construct with an explicit integer λ² value. This is the
    /// primary constructor — pass λ² directly (e.g. 13, 100, 1000).
    pub fn from_lambda_sq_integer(lambda_sq: u64, n_modes: usize) -> Self {
        Self {
            lambda_sq: LambdaSq::integer(lambda_sq),
            n_modes,
        }
    }

    /// Construct with an explicit fractional λ² value (e.g. 12.5, 2.7).
    /// HP uses the 18-significant-digit scientific decimal of this binary64
    /// value. See [`LambdaSq`] for the precision and near-one limitations.
    pub fn from_lambda_sq_fractional(lambda_sq: f64, n_modes: usize) -> Self {
        Self {
            lambda_sq: LambdaSq::fractional(lambda_sq),
            n_modes,
        }
    }

    /// The f64 value of λ² (for display, f64-tier computation).
    pub fn lambda_squared(&self) -> f64 {
        self.lambda_sq.value_f64
    }
    /// The integer floor of λ² — the prime cutoff for the Weil-form sum.
    pub fn lambda_sq_int(&self) -> u64 {
        self.lambda_sq.value_u64
    }
    /// `L = ln(λ²) = 2 ln λ` at f64.
    pub fn log_length(&self) -> f64 {
        self.lambda_sq.value_f64.ln()
    }
    /// Matrix dimension `2N+1`.
    ///
    /// # Panics
    /// Panics if the dimension cannot be represented by `usize`.
    pub fn matrix_size(&self) -> usize {
        self.n_modes
            .checked_mul(2)
            .and_then(|size| size.checked_add(1))
            .expect("CCM matrix dimension 2N+1 must fit usize")
    }

    /// Map a centered basis index `n ∈ [-N, N]` to the row/column
    /// position in the row-major matrix (`n = 0` → position `N`).
    ///
    /// # Panics
    /// Panics if the matrix dimension is unrepresentable or `n` is outside
    /// the centered basis.
    pub fn idx(&self, n: i64) -> usize {
        self.matrix_size();
        let magnitude =
            usize::try_from(n.unsigned_abs()).expect("CCM basis index magnitude must fit usize");
        assert!(
            magnitude <= self.n_modes,
            "CCM basis index must lie in [-N, N]"
        );
        if n < 0 {
            self.n_modes - magnitude
        } else {
            self.n_modes + magnitude
        }
    }
}

/// Result of a single CCM run at f64 precision.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CcmResult {
    /// Positive sign-change roots discovered from the even-sector state at f64
    /// precision, in ascending order. The search has no completeness guarantee.
    pub eigenvalues_pos: Vec<f64>,
    /// Lowest computed even-sector Ritz value. Binary64 does not certify its
    /// sign or its ordering against the odd sector or the continuum spectrum.
    pub weil_min_eigenvalue: f64,
    /// Selected even-sector state in centered V_n order, scaled so its
    /// component sum is sqrt(L). It is not normalized to unit ℓ² norm.
    pub xi: Vec<f64>,
    /// Wall-clock seconds for the entire f64 run.
    pub elapsed_seconds: f64,
}

impl CcmResult {
    /// Positive sign-change roots of the admitted binary64 stored state.
    ///
    /// These are distinct from eigenvalues of the Tau/Weil quadratic form.
    pub fn spectral_roots(&self) -> &[f64] {
        &self.eigenvalues_pos
    }
}

/// Enumerate prime powers `n = p^k` with `1 < n ≤ bound`.
///
/// Returns triples `(n, p, j)` where `n = p^j`. Callers are responsible
/// for computing `log p` themselves at the appropriate precision (HP via
/// `Float::with_val(prec, p).ln()`, f64 via `(p as f64).ln()`). This
/// keeps `prime_powers_up_to` precision-agnostic so HP code paths never
/// receive an f64-truncated logarithm. Panics on unrepresentable or unavailable
/// sieve/output capacity; use [`try_prime_powers_up_to`] to propagate errors.
pub fn prime_powers_up_to(bound: u64) -> Vec<(u64, u64, u32)> {
    try_prime_powers_up_to(bound).expect("representable and allocatable prime-power sieve required")
}

/// Checked prime-power enumeration. The integer multiplication is bounded
/// before evaluation; sieve and output capacity errors are propagated.
pub fn try_prime_powers_up_to(bound: u64) -> Result<Vec<(u64, u64, u32)>> {
    let mut out = Vec::new();
    for p in xc_numerics::primes::try_sieve_primes(bound)? {
        let mut q = p;
        let mut j = 1;
        while q <= bound {
            out.try_reserve(1)?;
            out.push((q, p, j));
            if q > bound / p {
                break;
            }
            q *= p;
            j += 1;
        }
    }
    out.sort_unstable_by_key(|&(n, _, _)| n);
    Ok(out)
}

/// Euler-Mascheroni constant γ ≈ 0.5772. Used in the archimedean
/// (gamma-factor) contribution W_R of the Weil quadratic form. Quoted
/// at full precision for documentation; f64 rounds to its nearest value.
#[allow(clippy::excessive_precision)]
pub const EULER_GAMMA: f64 = 0.5772156649015328606;

/// Numerical zero threshold for eigenvector sum normalization.
/// If |Σ ξ_j| < this value, the eigenvector is considered degenerate
/// and normalization is skipped with an error.
pub const EIGENVECTOR_SUM_THRESHOLD: f64 = 1e-14;

/// Singularity guard for the W_R integrand near x = 0.
/// When |x| < this value, the integrand uses its Taylor expansion
/// instead of the direct formula (which has a 0/0 form at x = 0).
pub const INTEGRAND_SINGULARITY_GUARD: f64 = 1e-9;

/// Default bisection tolerance for f64 spectrum root-finding.
pub const DEFAULT_BISECT_TOL: f64 = 1e-12;

/// Default maximum bisection iterations for f64 spectrum root-finding.
pub const DEFAULT_BISECT_MAX_ITER: usize = 200;

/// Binary64 exploratory even-sector computation. Matrix integration scales
/// with the highest Fourier mode. Near-degenerate states and unresolved boundary
/// sums return an error requesting the HP route; no certified root
/// ordinal, spectral sign, or continuum ground selection is supplied.
pub fn run_f64(params: &CcmParams) -> Result<CcmResult> {
    use nalgebra::SymmetricEigen;

    let start = Instant::now();
    let n = params.n_modes;
    let l = params.log_length();

    let tau = build_tau_f64(params, l, params.lambda_sq_int())?;

    // Project the operator onto an orthonormal even basis before solving.
    // Projecting the unrestricted minimum eigenvector afterward can turn an
    // odd vector's rounding noise into a spurious normalized even state.
    let mut even = nalgebra::DMatrix::<f64>::zeros(n + 1, n + 1);
    even[(0, 0)] = tau[(n, n)];
    for i in 1..=n {
        even[(0, i)] = (tau[(n, n + i)] + tau[(n, n - i)]) / 2.0_f64.sqrt();
        even[(i, 0)] = even[(0, i)];
        for j in 1..=n {
            even[(i, j)] = 0.5
                * (tau[(n + i, n + j)]
                    + tau[(n + i, n - j)]
                    + tau[(n - i, n + j)]
                    + tau[(n - i, n - j)]);
        }
    }
    if even.iter().any(|entry| !entry.is_finite()) {
        return Err(anyhow!("CCM even projection contains a nonfinite entry"));
    }
    let budget = 4096_usize
        .checked_mul(n + 1)
        .ok_or_else(|| anyhow!("CCM eigensolve budget overflow"))?;
    let mut eig = SymmetricEigen::try_new(even.clone(), f64::EPSILON, budget)
        .ok_or_else(|| anyhow!("CCM even eigensolve did not converge"))?;
    if eig
        .eigenvalues
        .iter()
        .chain(eig.eigenvectors.iter())
        .any(|entry| !entry.is_finite())
    {
        return Err(anyhow!("CCM even eigensolve returned nonfinite values"));
    }
    // The library QR can attach a tiny eigenvalue to another column; complete
    // the basis so the ascending values are paired with their own vectors.
    let values = xc_numerics::symmetric_f64::complete_symmetric_eigensystem_f64(
        even.as_slice(),
        n + 1,
        eig.eigenvectors.as_mut_slice(),
    )?;
    let eps_n = values[0];
    // This is an explicit numerical admission policy, not an assembly-error
    // certificate. A small residual alone cannot identify a vector within a
    // cluster. Reserve separation well above a dimension-scaled binary64
    // eigensolver floor before using the selected vector as a secular source.
    let spectral_scale = values.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    let relative_floor = 16.0 * (n + 1) as f64 * f64::EPSILON;
    let relative_gap = if values.len() == 1 {
        1.0
    } else if spectral_scale > 0.0 {
        values[1] / spectral_scale - values[0] / spectral_scale
    } else {
        0.0
    };
    if !relative_gap.is_finite() || relative_gap <= 1024.0 * relative_floor {
        return Err(anyhow!(
            "CCM even ground state unresolved in binary64: insufficient spectral separation; use the HP route"
        ));
    }
    let sector_state = eig.eigenvectors.column(0);
    let mut xi = vec![0.0_f64; 2 * n + 1];
    xi[n] = sector_state[0];
    for j in 1..=n {
        let value = sector_state[j] / 2.0_f64.sqrt();
        xi[n + j] = value;
        xi[n - j] = value;
    }

    // Normalize: Σ ξ_j = √L.
    let sum_xi: f64 = xi.iter().sum();
    let sum_abs: f64 = xi.iter().map(|x| x.abs()).sum();
    let sum_roundoff =
        (xi.len() as f64 * f64::EPSILON) / (1.0 - xi.len() as f64 * f64::EPSILON) * sum_abs;
    // Normalizing a small sum also amplifies state uncertainty. Use the
    // same numerical separation policy in this linear functional, whose
    // Euclidean operator norm is sqrt(full_dimension).
    let sum_state_uncertainty = (xi.len() as f64).sqrt() * relative_floor / relative_gap;
    if !sum_xi.is_finite()
        || sum_xi.abs()
            <= (32.0 * (sum_roundoff + sum_state_uncertainty)).max(EIGENVECTOR_SUM_THRESHOLD)
    {
        return Err(anyhow!(
            "CCM boundary sum unresolved in binary64; use the HP route"
        ));
    }
    let scale = l.sqrt() / sum_xi;
    for v in xi.iter_mut() {
        let nonzero = *v != 0.0;
        *v *= scale;
        if !v.is_finite() || (nonzero && *v == 0.0) {
            return Err(anyhow!("CCM state normalization is outside binary64 range"));
        }
    }

    let eigenvalues_pos =
        solve_spectrum_f64(&xi, n, l, DEFAULT_BISECT_TOL, DEFAULT_BISECT_MAX_ITER)?;

    Ok(CcmResult {
        eigenvalues_pos,
        weil_min_eigenvalue: eps_n,
        xi,
        elapsed_seconds: start.elapsed().as_secs_f64(),
    })
}

fn archimedean_origin_limit_f64(n: i64, m: i64, l: f64) -> f64 {
    let omega_0 = if n == m { 2.0 } else { 0.0 };
    // Both the diagonal cosine formula and the off-diagonal sine difference
    // have omega'(0) = -2/L. Since 2*sinh(x) = 2*x + O(x^3), the limit is
    // omega(0)/4 + omega'(0)/2, independent of the off-diagonal indices.
    let omega_prime_0 = -2.0 / l;
    omega_0 / 4.0 + omega_prime_0 / 2.0
}

fn build_tau_f64(params: &CcmParams, l: f64, lambda_sq_int: u64) -> Result<nalgebra::DMatrix<f64>> {
    use nalgebra::DMatrix;

    let n_max = params.n_modes;
    if !l.is_finite()
        || l <= 0.0
        || !params.lambda_squared().is_finite()
        || params.lambda_squared() <= 1.0
        || (params.lambda_sq.is_integer && params.lambda_squared() != lambda_sq_int as f64)
        || (!params.lambda_sq.is_integer && params.lambda_squared().floor() as u64 != lambda_sq_int)
    {
        return Err(anyhow!(
            "CCM requires finite lambda-squared > 1 and a matching prime cutoff"
        ));
    }
    let dim = n_max
        .checked_mul(2)
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| anyhow!("CCM matrix dimension overflow"))?;
    let count = dim
        .checked_mul(dim)
        .ok_or_else(|| anyhow!("CCM matrix size overflow"))?;
    let quadrature_order = n_max
        .checked_mul(3)
        .and_then(|v| v.checked_add(32))
        .ok_or_else(|| anyhow!("CCM quadrature order overflow"))?
        .max(64);
    // Resolve one rule for the entire matrix. Fixed GL64 aliases higher modes.
    let (nodes, weights) = xc_numerics::quadrature::try_gl_nodes_weights_f64(quadrature_order)?;
    let quadrature: Vec<(f64, f64)> = nodes
        .iter()
        .zip(&weights)
        .map(|(&node, &weight)| (0.5 * l * (1.0 + node), 0.5 * l * weight))
        .collect();
    let mut entries = Vec::new();
    entries.try_reserve_exact(count)?;
    entries.resize(count, 0.0);
    let mut tau = DMatrix::<f64>::from_vec(dim, dim, entries);

    let kappa = (4.0 * std::f64::consts::PI * (0.5 * l).tanh()).ln() + EULER_GAMMA;
    let sinh2_l_over_4 = (l / 4.0).sinh().powi(2);
    let sixteen_pi2 = 16.0 * std::f64::consts::PI * std::f64::consts::PI;
    let l2 = l * l;
    let prime_powers = try_prime_powers_up_to(lambda_sq_int)?;

    for n in -(n_max as i64)..=(n_max as i64) {
        for m in -(n_max as i64)..=(n_max as i64) {
            let nf = n as f64;
            let mf = m as f64;

            // W_{0,2}
            let num = l2 - sixteen_pi2 * mf * nf;
            let den = (l2 + sixteen_pi2 * mf * mf) * (l2 + sixteen_pi2 * nf * nf);
            let w02 = 32.0 * l * sinh2_l_over_4 * num / den;

            // W_R
            let omega_0 = if n == m { 2.0 } else { 0.0 };
            let two_pi_n_over_l = 2.0 * std::f64::consts::PI * nf / l;
            let two_pi_m_over_l = 2.0 * std::f64::consts::PI * mf / l;
            let integrand = |x: f64| -> f64 {
                if x == 0.0 {
                    return archimedean_origin_limit_f64(n, m, l);
                }
                if n == m {
                    let phase = two_pi_n_over_l * x;
                    let cosine = phase.cos();
                    // Rearrange exp(x/2)*(1-x/L)*cos(phase)-1 without
                    // subtracting nearly equal numbers at the origin.
                    (-2.0 * (0.5 * phase).sin().powi(2) - (x / l) * cosine
                        + (0.5 * x).exp_m1() * (1.0 - x / l) * cosine)
                        / x.sinh()
                } else {
                    let difference = std::f64::consts::PI * (nf - mf) * x / l;
                    let mean = std::f64::consts::PI * (nf + mf) * x / l;
                    let sinc = if difference == 0.0 {
                        1.0
                    } else {
                        difference.sin() / difference
                    };
                    let omega = -2.0 * (x / l) * sinc * mean.cos();
                    (0.5 * x).exp() * omega / (2.0 * x.sinh())
                }
            };
            let integral: f64 = quadrature
                .iter()
                .map(|&(x, weight)| weight * integrand(x))
                .sum();
            let wr = (omega_0 / 2.0) * kappa + integral;

            // W_p
            let mut wp_sum = 0.0_f64;
            for &(k, p, _j) in &prime_powers {
                let log_p = (p as f64).ln();
                let y = (k as f64).ln();
                let q = if n == m {
                    2.0 * (1.0 - y / l) * (two_pi_n_over_l * y).cos()
                } else {
                    ((two_pi_m_over_l * y).sin() - (two_pi_n_over_l * y).sin())
                        / (std::f64::consts::PI * ((n - m) as f64))
                };
                wp_sum += log_p * (k as f64).powf(-0.5) * q;
            }

            let entry = w02 - wr - wp_sum;
            if !entry.is_finite() {
                return Err(anyhow!("CCM matrix entry is not finite"));
            }
            tau[(params.idx(n), params.idx(m))] = entry;
        }
    }
    Ok(tau)
}

/// Search for positive sign-change roots of the even-state secular function
/// `R(t) = ξ_0 + 2t Σ ξ_j/(t-j²)`, returned as ascending physical ordinates.
///
/// Signed residues can place any number of roots in one pole gap or beyond the
/// last pole, so each window, including the one below the first pole and the
/// whole exterior, is subdivided using binary64 root-exclusion and
/// monotonicity estimates. This exploratory route does not certify completeness.
/// Roots that binary64 cannot
/// separate return an error rather than being dropped; a root within one
/// binary64 spacing of a pole is reported at that spacing.
/// Inputs must be finite, exactly even, and have nonzero component sum, as
/// required by the CCM quotient. A common nonzero state scale is normalized out.
/// `tol` is positive and controls absolute/relative t-bracket width; only a zero
/// of the exact stored-dyadic secular sum can short-circuit that test. Exhausted work,
/// nonfinite evaluations and unrepresentable physical ordinates return errors.
/// These point brackets do not certify physical-ordinate error or a comparison
/// against a Riemann zero. Signs and zeros use exact rational evaluation of the
/// supplied dyadic source under an exact common power-of-two scale; heuristic exclusion does not prove completeness.
/// More than 4096 active poles exceeds the explicit work budget. For certified
/// counts use HP certified discovery.
pub fn solve_spectrum_f64(
    xi: &[f64],
    n_max: usize,
    l: f64,
    tol: f64,
    max_iter: usize,
) -> Result<Vec<f64>> {
    native_secular::solve(xi, n_max, l, tol, max_iter)
}

// Reference Riemann-zero literals below are quoted at published precision
// (more digits than f64 holds); the excess is harmless on parse.
#[cfg(test)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;

    #[test]
    fn archimedean_origin_limit_matches_direct_integrand_extrapolation() {
        for cutoff in [2.0_f64, 7.0, 13.0, 1200.0] {
            let l = cutoff.ln();
            for (n, m) in [
                (0, 0),
                (1, 1),
                (-3, -3),
                (1, 0),
                (2, 1),
                (3, 1),
                (3, 2),
                (0, 1),
                (-1, 1),
                (3, -2),
            ] {
                // Extrapolate the original analytic integrand, never calling
                // its origin guard or substituting the proposed limit.
                let direct = |x: f64| {
                    let a = 2.0 * std::f64::consts::PI * n as f64 / l;
                    let b = 2.0 * std::f64::consts::PI * m as f64 / l;
                    let omega = if n == m {
                        2.0 * (1.0 - x / l) * (a * x).cos()
                    } else {
                        ((b * x).sin() - (a * x).sin()) / (std::f64::consts::PI * (n - m) as f64)
                    };
                    ((x / 2.0).exp() * omega - if n == m { 2.0 } else { 0.0 }) / (2.0 * x.sinh())
                };
                let h = l * 2.0_f64.powi(-20);
                let extrapolated = 3.0 * direct(h) - 3.0 * direct(2.0 * h) + direct(3.0 * h);
                let limit = archimedean_origin_limit_f64(n, m, l);
                assert!(
                    (limit - extrapolated).abs() < 1e-8 * (1.0 + limit.abs()),
                    "C={cutoff}, n={n}, m={m}: guard={limit}, extrapolated={extrapolated}"
                );
            }
        }
    }

    #[test]
    fn archimedean_origin_guard_is_inactive_for_integer_but_reachable_for_fractional_cutoffs() {
        let (nodes, _) = xc_numerics::quadrature::gl_nodes_weights_f64(64);
        let minimum = |p: CcmParams| {
            nodes
                .iter()
                .map(|x| 0.5 * p.log_length() * (1.0 + x))
                .fold(f64::INFINITY, f64::min)
        };
        // C=2 is the smallest valid integer cutoff; mapped nodes increase with C.
        assert!(minimum(CcmParams::from_lambda_sq_integer(2, 1)) > INTEGRAND_SINGULARITY_GUARD);
        assert!(
            minimum(CcmParams::from_lambda_sq_fractional(1.00000001, 1))
                < INTEGRAND_SINGULARITY_GUARD
        );
    }

    #[test]
    fn f64_matrix_matches_independent_high_mode_and_small_length_integrals() {
        // Independently integrated defining Weil distribution at 90 decimal
        // digits with mpmath tanh-sinh, including direct pole integration.
        // Reproduction: Research 2026-09-23-full-mathematics-revalidation.
        let params = CcmParams::from_lambda_sq_integer(13, 120);
        let matrix = build_tau_f64(&params, params.log_length(), 13).unwrap();
        for (n, m, expected) in [
            (64, 64, 4.182967328799878),
            (64, 63, 0.31200771112145644),
            (120, 120, 4.636760801387921),
            (120, 0, 0.00670592132993459),
        ] {
            let actual = matrix[(params.idx(n), params.idx(m))];
            assert!(
                (actual - expected).abs() < 2e-11,
                "n={n}, m={m}: actual={actual}, reference={expected}"
            );
        }
        // The reference uses the exact binary64 cutoff input, not a rounded
        // decimal surrogate. The former fixed origin guard is inaccurate here.
        let params = CcmParams::from_lambda_sq_fractional(1.00000001, 3);
        let matrix = build_tau_f64(&params, params.log_length(), 1).unwrap();
        for (n, m, expected) in [(3, 3, 19.522062402308804), (3, 2, 0.004117773807046028)] {
            let actual = matrix[(params.idx(n), params.idx(m))];
            assert!(
                (actual - expected).abs() < 2e-11,
                "small L n={n}, m={m}: actual={actual}, reference={expected}"
            );
        }
    }

    #[test]
    fn native_even_ritz_values_match_independent_defining_integral_matrices() {
        let oracle: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/fixtures/native_ccm_oracle.json"))
                .unwrap();
        let mut checked = 0;
        for case in oracle["ritz_cases"].as_array().unwrap() {
            let cutoff = case["cutoff"].as_u64().unwrap();
            let modes = case["modes"].as_u64().unwrap() as usize;
            let params = CcmParams::from_lambda_sq_integer(cutoff, modes);
            let tau = build_tau_f64(&params, params.log_length(), cutoff).unwrap();
            // Form the orthonormal reflection embedding independently of
            // run_f64, so rejected state normalization cannot skip an oracle.
            let mut q = nalgebra::DMatrix::<f64>::zeros(2 * modes + 1, modes + 1);
            q[(modes, 0)] = 1.0;
            for j in 1..=modes {
                q[(modes - j, j)] = 1.0 / 2.0_f64.sqrt();
                q[(modes + j, j)] = 1.0 / 2.0_f64.sqrt();
            }
            let even = q.transpose() * tau * q;
            let eig = nalgebra::SymmetricEigen::new(even);
            let minimum = eig
                .eigenvalues
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            let expected = case["even_minimum"]
                .as_str()
                .unwrap()
                .parse::<f64>()
                .unwrap();
            assert!(
                (minimum - expected).abs() < 2e-12 * (1.0 + expected.abs()),
                "C={cutoff} N={modes}: matrix eigenvalue={minimum}, reference={expected}"
            );
            checked += 1;
        }
        assert_eq!(checked, 24);
    }

    #[test]
    fn f64_selected_even_state_has_a_small_full_matrix_residual() {
        // This source has a resolved even state in binary64. The former
        // cutoff-13/N=32 fixture has an unresolved gap and is checked below.
        let params = CcmParams::from_lambda_sq_integer(2, 20);
        let result = run_f64(&params).unwrap();
        let matrix = build_tau_f64(&params, params.log_length(), 2).unwrap();
        let xi = nalgebra::DVector::from_column_slice(&result.xi);
        let residual = &matrix * &xi - result.weil_min_eigenvalue * &xi;
        assert!(residual.norm() / xi.norm() < 1e-11);
        for j in 0..=params.n_modes {
            assert_eq!(result.xi[params.n_modes + j], result.xi[params.n_modes - j]);
        }
        let unresolved = CcmParams::from_lambda_sq_integer(13, 32);
        let error = run_f64(&unresolved).unwrap_err();
        assert!(error.to_string().contains("unresolved in binary64"));
    }

    #[test]
    fn f64_secular_search_includes_the_first_positive_gap() {
        // R(t)=(3t-1)/(t-1), with the exact positive ordinate 1/sqrt(3)
        // when L=2*pi. No reference-zero input or production root oracle.
        let roots = solve_spectrum_f64(&[1.0, 1.0, 1.0], 1, 2.0 * std::f64::consts::PI, 1e-14, 200)
            .unwrap();
        assert_eq!(roots.len(), 1);
        assert!((roots[0] - 1.0 / 3.0_f64.sqrt()).abs() < 1e-13);
        assert!(solve_spectrum_f64(&[f64::NAN; 3], 1, 1.0, 1e-14, 200).is_err());
        assert!(solve_spectrum_f64(&[1.0], 1, 1.0, 1e-14, 200).is_err());
    }

    /// Solve with L = 2π, so each returned ordinate is √t.
    fn secular_roots_in_t(positive: &[f64]) -> Vec<f64> {
        let n = positive.len() - 1;
        let xi = positive[1..]
            .iter()
            .rev()
            .chain(positive)
            .copied()
            .collect::<Vec<_>>();
        solve_spectrum_f64(&xi, n, 2.0 * std::f64::consts::PI, 1e-14, 200)
            .unwrap()
            .iter()
            .map(|ordinate| ordinate * ordinate)
            .collect()
    }

    fn assert_roots(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len(), "{actual:?} vs {expected:?}");
        for (a, e) in actual.iter().zip(expected) {
            assert!(
                (a - e).abs() <= 1e-10 * e.abs(),
                "{actual:?} vs {expected:?}"
            );
        }
    }

    #[test]
    fn f64_secular_search_finds_two_roots_in_one_pole_gap() {
        // R(t) = -10 + 1/(t-1) - 1/(t-4): both gap ends are +inf, so the
        // former one-bisection-per-gap search found no bracket and dropped
        // both roots of t^2 - 5t + 4.3 = 0.
        let roots = secular_roots_in_t(&[-10.75, 0.5, -0.125]);
        let d = 7.8_f64.sqrt();
        assert_roots(&roots, &[(5.0 - d) / 2.0, (5.0 + d) / 2.0]);
    }

    #[test]
    fn f64_secular_search_finds_every_exterior_root() {
        // R(t) = 1 - 20/(t-1) + 1/(t-4): +inf after the last pole and 1 at
        // infinity, with two roots t = 12 -/+ sqrt(61) beyond the last pole.
        let roots = secular_roots_in_t(&[20.75, -10.0, 0.125]);
        let d = 61.0_f64.sqrt();
        assert_roots(&roots, &[12.0 - d, 12.0 + d]);
    }

    #[test]
    fn f64_secular_search_handles_zero_constant_and_tiny_residues() {
        // xi_0 = 0: R(t) = t(2/(t-1) + 2/(t-4)); t = 0 is not positive.
        assert_roots(&secular_roots_in_t(&[0.0, 1.0, 1.0]), &[2.5]);
        // R(t) = 1 + 1/(t-1) - 1e-6/(t-4) has roots t^2 - (4+1e-6)t + 1e-6 = 0,
        // one just below 1 in the first window and one hugging the weak pole.
        let roots = secular_roots_in_t(&[2.5e-7, 0.5, -1.25e-7]);
        let (b, c) = (4.0_f64 + 1e-6, 1e-6);
        let d = (b * b - 4.0 * c).sqrt();
        let large = (b + d) / 2.0;
        assert_eq!(roots.len(), 2, "{roots:?}");
        assert!((roots[0] - c / large).abs() < 1e-13, "{roots:?}");
        assert!((roots[1] - large).abs() < 1e-12, "{roots:?}");
    }

    #[test]
    fn f64_ccm_rejects_an_unresolved_boundary_normalization() {
        // C=5,N=20 once exposed a mismatched QR column. Correct column pairing
        // alone is insufficient: the independent finite-root oracle still
        // shows large tail errors when its boundary sum is unresolved.
        let error = run_f64(&CcmParams::from_lambda_sq_integer(5, 20))
            .expect_err("binary64 must reject an unresolved source");
        assert!(error.to_string().contains("unresolved in binary64"));
    }

    #[test]
    fn f64_ccm_rejects_unresolved_states_before_root_discovery() {
        for modes in [8, 20] {
            let error = run_f64(&CcmParams::from_lambda_sq_integer(13, modes))
                .expect_err("unresolved state must not become a root source");
            assert!(error.to_string().contains("unresolved in binary64"));
        }
        let result = run_f64(&CcmParams::from_lambda_sq_integer(2, 20)).unwrap();
        assert!(!result.eigenvalues_pos.is_empty());
        assert!(result.eigenvalues_pos.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn f64_ccm_rejects_invalid_cutoffs_before_assembly() {
        for cutoff in [0.0, 0.5, 1.0, f64::NAN, f64::INFINITY] {
            assert!(run_f64(&CcmParams::from_lambda_sq_fractional(cutoff, 1)).is_err());
        }
    }

    #[test]
    fn f64_ccm_result_round_trips_without_loss() {
        let result = CcmResult {
            eigenvalues_pos: vec![14.134725141734695, 21.022039638771556],
            weil_min_eigenvalue: 1.0e-200,
            xi: vec![-0.125, 0.75],
            elapsed_seconds: 1.25,
        };
        let encoded = serde_json::to_vec(&result).unwrap();
        let decoded: CcmResult = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, result);
    }

    /// Prime powers up to 13 should include {2, 3, 4, 5, 7, 8, 9, 11, 13}.
    /// Each entry is (k, p, j) with k = p^j.
    #[test]
    fn prime_powers_up_to_13() {
        let pp = prime_powers_up_to(13);
        let ns: Vec<u64> = pp.iter().map(|&(n, _, _)| n).collect();
        assert!(ns.contains(&2));
        assert!(ns.contains(&3));
        assert!(ns.contains(&4)); // 2²
        assert!(ns.contains(&5));
        assert!(ns.contains(&7));
        assert!(ns.contains(&8)); // 2³
        assert!(ns.contains(&9)); // 3²
        assert!(ns.contains(&11));
        assert!(ns.contains(&13));
        assert!(!ns.contains(&6)); // not a prime power
        assert!(!ns.contains(&10));

        // Check (p, j) is correct for k = p^j.
        for &(k, p, j) in &pp {
            let mut prod: u64 = 1;
            for _ in 0..j {
                prod *= p;
            }
            assert_eq!(k, prod, "k = p^j must hold; got k={}, p={}, j={}", k, p, j);
        }
    }

    /// CcmParams basic properties.
    #[test]
    fn ccm_params_basic() {
        let p = CcmParams::from_lambda_sq_integer(13, 120);
        assert!((p.lambda_squared() - 13.0).abs() < 1e-12);
        assert_eq!(p.matrix_size(), 241);
        assert_eq!(p.idx(0), 120);
        assert_eq!(p.idx(-120), 0);
        assert_eq!(p.idx(120), 240);
    }

    /// f64 tier should produce a non-empty spectrum with reasonable
    /// structure at λ²=13, N=120. Note: f64 tier uses bisection on R(t)
    /// without Newton-from-seed, so eigenvalues are Weil-form roots
    /// rather than Riemann zeros (those need the HP path).
    ///
    /// Skipped in debug builds: the unoptimized nalgebra `SymmetricEigen`
    /// on a 241×241 matrix allocates a stack frame for every intermediate
    /// value, exhausting even a large stack before finishing.
    ///
    /// Skipped always (ignored): building the 241×241 τ-matrix via
    /// O(N²) GL integrations is a multi-minute test even in release mode;
    /// run explicitly with `--include-ignored` when validating the f64 path.
    #[test]
    #[ignore = "f64 τ-matrix build (58K GL integrations) — too slow for routine test runs; use --include-ignored to run explicitly"]
    #[cfg(not(debug_assertions))]
    fn run_f64_produces_spectrum() {
        let params = CcmParams::from_lambda_sq_integer(13, 120);
        let result = run_f64(&params).unwrap();
        assert!(
            !result.eigenvalues_pos.is_empty(),
            "should produce at least one eigenvalue"
        );
        // ε_N should be tiny at λ²=13, N=120 even at f64.
        assert!(
            result.weil_min_eigenvalue.abs() < 1e-10,
            "ε_N = {} should be tiny",
            result.weil_min_eigenvalue
        );
        // ξ should be normalized (Σ ξ_j = √L).
        let sum_xi: f64 = result.xi.iter().sum();
        let l = 13.0_f64.ln();
        let expected_sum = l.sqrt();
        assert!(
            (sum_xi - expected_sum).abs() < 1e-3,
            "Σ ξ_j = {} should be √L = {}",
            sum_xi,
            expected_sum
        );
        // Eigenvalues should be sorted ascending.
        for w in result.eigenvalues_pos.windows(2) {
            assert!(w[0] < w[1], "eigenvalues should be ascending");
        }
        // Some eigenvalue should match the 2nd Riemann zero (21.02)
        // since it's outside the dense Weil-form region.
        let target = 21.022040;
        let closest_to_2nd = result
            .eigenvalues_pos
            .iter()
            .map(|&e| (e - target).abs())
            .fold(f64::INFINITY, f64::min);
        assert!(
            closest_to_2nd < 0.01,
            "no eigenvalue near 21.02 (closest: {:.4e})",
            closest_to_2nd
        );
    }

    /// solve_spectrum_f64 with a synthetic ξ vector: constant ξ_0 = 1, all
    /// others zero. R(t) = 1 + 0 = 1 everywhere, so no roots exist.
    /// This tests the "no root found" path.
    #[test]
    fn solve_spectrum_constant_xi_no_roots() {
        let n_max = 5;
        let l = 13.0_f64.ln(); // L = ln(13)
        let mut xi = vec![0.0_f64; 2 * n_max + 1];
        xi[n_max] = 1.0; // only ξ_0 nonzero
        let roots =
            solve_spectrum_f64(&xi, n_max, l, DEFAULT_BISECT_TOL, DEFAULT_BISECT_MAX_ITER).unwrap();
        // With only ξ_0 nonzero, R(t) = ξ_0 = 1 (constant), no zeros.
        assert!(
            roots.is_empty(),
            "constant R(t) should have no roots, got {:?}",
            roots
        );
    }

    /// solve_spectrum_f64 with a simple ξ that has known structure.
    /// If ξ_0 = 0 and ξ_1 = ξ_{-1} = 1 (all others zero), then
    /// R(t) = 0 + 2t · [1/(t - 1)] = 2t/(t-1).
    /// This has a zero at t = 0 (outside our search range) and no zero
    /// in (1, 4), (4, 9), etc. — the function is always positive for t > 1.
    /// So we expect no roots in the standard intervals.
    #[test]
    fn solve_spectrum_single_mode() {
        let n_max = 3;
        let l = 13.0_f64.ln();
        let mut xi = vec![0.0_f64; 2 * n_max + 1];
        xi[n_max + 1] = 1.0; // an unpaired mode is not an even state
        assert!(
            solve_spectrum_f64(&xi, n_max, l, DEFAULT_BISECT_TOL, DEFAULT_BISECT_MAX_ITER).is_err()
        );
        xi[n_max - 1] = 1.0; // reflected pair: ξ_1 = ξ_{-1} = 1
        let roots =
            solve_spectrum_f64(&xi, n_max, l, DEFAULT_BISECT_TOL, DEFAULT_BISECT_MAX_ITER).unwrap();
        // R(t) = 2t/(t-1) for t > 1 is always positive, so no sign changes.
        assert!(
            roots.is_empty(),
            "single-mode R(t) should have no roots in standard intervals"
        );
    }

    /// CcmParams accessor methods.
    #[test]
    fn ccm_params_accessors() {
        let p = CcmParams::from_lambda_sq_integer(100, 50);
        assert!((p.lambda_squared() - 100.0).abs() < 1e-12);
        assert!((p.log_length() - 100.0_f64.ln()).abs() < 1e-10);
    }

    /// prime_powers_up_to edge cases.
    #[test]
    fn prime_powers_edge_cases() {
        assert!(prime_powers_up_to(0).is_empty());
        assert!(prime_powers_up_to(1).is_empty());
        let pp2 = prime_powers_up_to(2);
        assert_eq!(pp2.len(), 1);
        assert_eq!(pp2[0].0, 2);
    }

    /// prime_powers_up_to output should be sorted ascending by k.
    #[test]
    fn prime_powers_output_is_sorted() {
        let pp = prime_powers_up_to(50);
        for w in pp.windows(2) {
            assert!(
                w[0].0 < w[1].0,
                "prime_powers_up_to should be sorted; {} >= {}",
                w[0].0,
                w[1].0
            );
        }
    }

    /// prime_powers_up_to: every (k, p, j) triple satisfies k = p^j exactly.
    #[test]
    fn prime_powers_triple_invariant() {
        for &(k, p, j) in &prime_powers_up_to(100) {
            let expected: u64 = (0..j).fold(1, |acc, _| acc * p);
            assert_eq!(k, expected, "triple ({}, {}, {}): k ≠ p^j", k, p, j);
        }
    }

    /// CcmParams::from_lambda_sq_integer(0) should not panic.
    /// lambda_sq_int will be 0 and prime_powers_up_to(0) returns empty.
    #[test]
    fn ccm_params_lambda_zero() {
        let p = CcmParams::from_lambda_sq_integer(0, 10);
        assert_eq!(p.lambda_sq_int(), 0);
        assert!(prime_powers_up_to(p.lambda_sq_int()).is_empty());
    }

    /// solve_spectrum_f64 with n_max=0 (degenerate ξ = [ξ_0]) should
    /// return an empty root list without panicking — there are no poles in
    /// R(t) and the bisection loop has no intervals to search.
    #[test]
    fn solve_spectrum_n_max_zero() {
        let l = 13.0_f64.ln();
        let xi = vec![1.0_f64]; // only ξ_0
        let roots =
            solve_spectrum_f64(&xi, 0, l, DEFAULT_BISECT_TOL, DEFAULT_BISECT_MAX_ITER).unwrap();
        assert!(roots.is_empty(), "n_max=0 should produce no roots");
    }
}

#[cfg(feature = "hp")]
pub mod state_geometry;

#[cfg(feature = "hp")]
pub mod retained_evidence;

/// Additional retained-source observations and external runtime research inputs.
#[cfg(feature = "hp")]
pub mod extended_research;

/// Complete retained-run convergence measurements and optional numerical models.
#[cfg(feature = "hp")]
pub mod convergence_capture;

#[cfg(feature = "hp")]
pub mod capture_runtime;
#[cfg(feature = "hp")]
pub mod research_completion;

#[cfg(feature = "hp")]
mod transform_enclosure;

#[cfg(feature = "hp")]
pub mod research_cohort;
#[cfg(feature = "hp")]
pub mod research_prepare;
#[cfg(feature = "hp")]
mod research_prepare_math;
#[cfg(feature = "hp")]
mod research_target;

#[cfg(feature = "hp")]
pub mod atom_research;

#[cfg(feature = "hp")]
mod band_runtime;

#[cfg(feature = "hp")]
mod research_export;
