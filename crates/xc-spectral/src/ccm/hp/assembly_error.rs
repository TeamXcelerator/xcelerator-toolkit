//! Rigorous distance between a stored CCM matrix and the exact finite Weil form.
//!
//! The exact form is enclosed by the cutoff-free closed-form assembly (FLINT/Arb
//! special functions and outward MPFR arithmetic) at the exact declared cutoff,
//! integer or fractional. Comparing it entrywise with the stored matrix bounds
//! the total quadrature, rounding and component error. By Weyl's inequality a
//! symmetric perturbation of spectral norm at most epsilon moves every
//! eigenvalue by at most epsilon, which turns each retained stored-matrix
//! eigenvalue enclosure into an exact-finite-form enclosure.
use super::*;
use rug::float::Round;
use xc_numerics::mpfr_interval::MpfrInterval;

pub const ASSEMBLY_ERROR_SEMANTICS: &str = "ccm-assembly-error-v0.16.0-v2";
const ASSEMBLY_ERROR_SCOPE: &str = "rigorous bounds on stored-matrix minus exact finite CCM Weil form at the declared cutoff, and exact-finite-form eigenvalue enclosures by Weyl's inequality; continuum limits are not addressed";
const BOUND_BITS: u32 = 64;

/// Upward-rounded norms of an entrywise error bound matrix.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CcmErrorNorms {
    pub max_entry_upper: String,
    pub frobenius_upper: String,
    pub row_sum_upper: String,
    /// min(Frobenius, row sum): a rigorous spectral-norm bound for a
    /// symmetric error matrix.
    pub spectral_upper: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CcmComponentErrors {
    pub pole: CcmErrorNorms,
    pub archimedean: CcmErrorNorms,
    pub prime: CcmErrorNorms,
}

/// One eigenvalue-type observable with its stored and exact-form enclosures.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CcmExactFormBound {
    pub observable: String,
    pub stored_lower: String,
    pub stored_upper: String,
    pub assembly_allowance: String,
    pub exact_form_lower: String,
    pub exact_form_upper: String,
    /// Decimal digits fixed by the exact-form enclosure, when it excludes zero.
    pub exact_form_resolved_digits: Option<String>,
    /// `stored` when the stored enclosure width dominates, else `assembly`.
    pub limiting_budget: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CcmAssemblyErrorAnalysis {
    pub schema_version: u32,
    pub semantics: String,
    pub claim_scope: String,
    pub lambda_squared: String,
    pub prime_cutoff: u64,
    pub n_modes: usize,
    pub precision_bits: u32,
    pub analysis_precision_bits: u32,
    pub geometric_terms: usize,
    pub tau_content_digest: String,
    pub matrix_scale: String,
    pub maximum_enclosure_width: String,
    pub full: CcmErrorNorms,
    pub even_sector: CcmErrorNorms,
    pub odd_sector: Option<CcmErrorNorms>,
    pub components: Option<CcmComponentErrors>,
    /// Entry `n` bounds the error in rows `+n` and `-n` (maximum entry).
    pub mode_profile: Vec<String>,
    /// floor(-log10(full spectral bound / matrix scale)).
    pub surviving_digits: Option<i64>,
    pub exact_form_bounds: Vec<CcmExactFormBound>,
    /// Exact-form root enclosures, absent with a stated limitation when the
    /// retained state or its spectral separation is unavailable.
    pub exact_form_roots: Option<CcmExactFormRoots>,
    pub exact_form_roots_limitation: Option<String>,
    pub outcome: String,
}

/// Enclosure of one computed root of the exact finite-form secular function.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CcmExactFormRoot {
    pub ordinal: usize,
    pub computed_value: String,
    /// `enclosed` or `unresolved`.
    pub outcome: String,
    pub exact_form_lower: Option<String>,
    pub exact_form_upper: Option<String>,
    pub exact_form_resolved_digits: Option<String>,
    pub reason: Option<String>,
}

/// Propagation of the assembly bound to the retained state and its roots.
/// sin(theta) <= (residual + eps) / (lambda_1,exact,lower - mu) bounds the
/// angle to the exact even ground state (Davis-Kahan, residual form); the
/// weights and exact poles then form an interval secular function whose
/// interval-Newton enclosures contain the exact-form roots.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CcmExactFormRoots {
    pub state_residual_upper: String,
    pub separation_lower: String,
    pub eigenvector_sine_upper: String,
    pub vector_distance_upper: String,
    pub rows: Vec<CcmExactFormRoot>,
}

fn up(value: Float) -> Float {
    Float::with_val_round(BOUND_BITS, &value, Round::Up).0
}

fn add_up(left: &Float, right: &Float) -> Float {
    Float::with_val_round(BOUND_BITS, left + right, Round::Up).0
}

fn mul_up(left: &Float, right: &Float) -> Float {
    Float::with_val_round(BOUND_BITS, left * right, Round::Up).0
}

fn decimal_up(value: &Float) -> String {
    value.to_string_radix_round(10, Some(20), Round::Up)
}

fn decimal_down(value: &Float) -> String {
    value.to_string_radix_round(10, Some(20), Round::Down)
}

/// Endpoint rounded outward with enough digits to keep the enclosure that the
/// binary endpoint carries.
fn endpoint_down(value: &Float) -> String {
    let digits = xc_numerics::reduction::roundtrip_decimal_digits(value.prec());
    value.to_string_radix_round(10, Some(digits), Round::Down)
}

fn endpoint_up(value: &Float) -> String {
    let digits = xc_numerics::reduction::roundtrip_decimal_digits(value.prec());
    value.to_string_radix_round(10, Some(digits), Round::Up)
}

/// Lower bound on the resolved decimal digits, -log10(width / scale), of the
/// exported decimal interval [lower, upper]. Endpoints are read outward and the
/// logarithm is rounded so the reported value never exceeds the true one.
fn resolved_digits(lower: &str, upper: &str, precision: u32) -> Option<String> {
    let parse = |text: &str, round: Round| {
        Float::parse(text)
            .ok()
            .map(|parsed| Float::with_val_round(precision, parsed, round).0)
    };
    let low = parse(lower, Round::Down)?;
    let high = parse(upper, Round::Up)?;
    if low.is_zero() || high.is_zero() || low.is_sign_positive() != high.is_sign_positive() {
        return None;
    }
    let width = Float::with_val_round(precision, &high - &low, Round::Up).0;
    let scale =
        Float::with_val(precision, low.abs_ref()).min(&Float::with_val(precision, high.abs_ref()));
    let mut ratio = Float::with_val_round(precision, &width / &scale, Round::Up).0;
    if ratio.is_zero() {
        return None;
    }
    ratio.log10_round(Round::Up);
    Some(decimal_down(&-ratio))
}

/// Upper bound of |interval - point|.
fn distance_upper(interval: &MpfrInterval, point: &Float) -> Float {
    let q = interval.precision().max(point.prec());
    let below = Float::with_val_round(q, interval.lower() - point, Round::Down)
        .0
        .abs();
    let above = Float::with_val_round(q, interval.upper() - point, Round::Up)
        .0
        .abs();
    up(if below > above { below } else { above })
}

#[derive(Clone)]
struct NormAccumulator {
    max: Float,
    squares: Float,
    rows: Vec<Float>,
}

impl NormAccumulator {
    fn new(dimension: usize) -> Self {
        let zero = Float::with_val(BOUND_BITS, 0);
        Self {
            max: zero.clone(),
            squares: zero.clone(),
            rows: vec![zero; dimension],
        }
    }
    fn add(&mut self, row: usize, column: usize, bound: &Float) {
        if bound > &self.max {
            self.max = bound.clone();
        }
        let square = mul_up(bound, bound);
        let weight = if row == column { 1 } else { 2 };
        for _ in 0..weight {
            self.squares = add_up(&self.squares, &square);
        }
        self.rows[row] = add_up(&self.rows[row], bound);
        if row != column {
            self.rows[column] = add_up(&self.rows[column], bound);
        }
    }
    fn norms(&self) -> CcmErrorNorms {
        let frobenius = Float::with_val_round(BOUND_BITS, self.squares.sqrt_ref(), Round::Up).0;
        let row_sum = self
            .rows
            .iter()
            .max_by(|a, b| a.total_cmp(b))
            .cloned()
            .unwrap_or_else(|| Float::with_val(BOUND_BITS, 0));
        let spectral = if frobenius < row_sum {
            frobenius.clone()
        } else {
            row_sum.clone()
        };
        CcmErrorNorms {
            max_entry_upper: decimal_up(&self.max),
            frobenius_upper: decimal_up(&frobenius),
            row_sum_upper: decimal_up(&row_sum),
            spectral_upper: decimal_up(&spectral),
        }
    }
}

fn parse_bound(text: &str) -> Result<Float> {
    Ok(Float::with_val_round(BOUND_BITS, Float::parse(text)?, Round::Up).0)
}

/// Entrywise sector error bounds from full-matrix error bounds `a`, plus the
/// correct-rounding error of each stored parity entry (at most |S|*2^-p).
fn sector_norms(
    a: &[Float],
    n: usize,
    stored: &[Float],
    p: u32,
    odd: bool,
) -> Result<CcmErrorNorms> {
    let full = 2 * n + 1;
    let at = |row: i64, column: i64| -> &Float {
        let r = (row + n as i64) as usize;
        let c = (column + n as i64) as usize;
        &a[r * full + c]
    };
    let offset = usize::from(odd);
    let dimension = n + 1 - offset;
    if stored.len() != dimension * dimension {
        bail!("stored parity matrix shape differs from the source");
    }
    // 0.70710678118654753 rounded up exceeds 1/sqrt(2).
    let inverse_sqrt2 = parse_bound("0.70710678118654753")?;
    let half = Float::with_val(BOUND_BITS, 0.5);
    let rounding = Float::with_val(BOUND_BITS, 1) >> p;
    let mut accumulator = NormAccumulator::new(dimension);
    for row in 0..dimension {
        for column in row..dimension {
            let k = (row + offset) as i64;
            let j = (column + offset) as i64;
            let transform = if !odd && k == 0 && j == 0 {
                at(0, 0).clone()
            } else if !odd && k == 0 {
                mul_up(&add_up(at(0, j), at(0, -j)), &inverse_sqrt2)
            } else {
                let sum = add_up(&add_up(at(k, j), at(k, -j)), &add_up(at(-k, j), at(-k, -j)));
                mul_up(&sum, &half)
            };
            let entry = Float::with_val(BOUND_BITS, stored[row * dimension + column].abs_ref());
            let bound = add_up(&transform, &mul_up(&up(entry), &rounding));
            accumulator.add(row, column, &bound);
        }
    }
    Ok(accumulator.norms())
}

fn exact_form_bound(
    observable: &str,
    lower: &Float,
    upper: &Float,
    allowance: &Float,
) -> CcmExactFormBound {
    let work = lower.prec().max(upper.prec()).max(BOUND_BITS);
    let exact_lower = Float::with_val_round(work, lower - allowance, Round::Down).0;
    let exact_upper = Float::with_val_round(work, upper + allowance, Round::Up).0;
    let stored_width = Float::with_val_round(work, upper - lower, Round::Up).0;
    let assembly_width = Float::with_val_round(work, allowance * 2u32, Round::Up).0;
    let exact_form_lower = endpoint_down(&exact_lower);
    let exact_form_upper = endpoint_up(&exact_upper);
    let digits = resolved_digits(&exact_form_lower, &exact_form_upper, work);
    CcmExactFormBound {
        observable: observable.to_owned(),
        stored_lower: endpoint_down(lower),
        stored_upper: endpoint_up(upper),
        assembly_allowance: decimal_up(allowance),
        exact_form_lower,
        exact_form_upper,
        exact_form_resolved_digits: digits,
        limiting_budget: if stored_width >= assembly_width {
            "stored"
        } else {
            "assembly"
        }
        .to_owned(),
    }
}

/// Inputs for exact-form eigenvalue enclosures. Every field is optional:
/// absent sources simply produce fewer bounds.
pub(super) struct ExactFormSources<'a> {
    pub primary: Option<&'a HighPrecResult>,
    pub sectors: Option<&'a CcmSectorGapHp>,
}

/// Compute the analysis. `components` supplies the stored pole, archimedean
/// and prime matrices for the per-component breakdown when available.
#[cfg(feature = "arb")]
pub(super) fn analyze_assembly_error(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    tau: &[Float],
    tau_digest: &ContentDigest,
    components: Option<&ComputedCcmMatrixComponents>,
    sources: ExactFormSources<'_>,
) -> Result<CcmAssemblyErrorAnalysis> {
    let n = params.n_modes;
    let p = cfg.precision_bits;
    let dimension = params.matrix_size();
    if tau.len() != dimension * dimension || !matrix_is_exactly_symmetric(tau, dimension) {
        bail!("assembly error analysis requires the exact stored symmetric Tau matrix");
    }
    let cutoff = lambda_squared_cache_identity(params);
    let analysis_bits = p.saturating_add(64);
    let resolved = super::super::cutoff_free::resolve_cutoff(&cutoff, n, analysis_bits)?;
    let zero = Float::with_val(BOUND_BITS, 0);
    let mut bounds = vec![zero.clone(); dimension * dimension];
    let mut full = NormAccumulator::new(dimension);
    let mut pole = NormAccumulator::new(dimension);
    let mut archimedean = NormAccumulator::new(dimension);
    let mut prime = NormAccumulator::new(dimension);
    let mut widest = zero.clone();
    let mut scale = zero;
    super::super::cutoff_free::visit_cells(
        &resolved.c,
        resolved.prime_cutoff,
        n,
        resolved.geometric_terms,
        |row, column, w02, wr, wp, exact| {
            let index = row * dimension + column;
            let bound = distance_upper(exact, &tau[index]);
            bounds[index] = bound.clone();
            bounds[column * dimension + row] = bound.clone();
            full.add(row, column, &bound);
            let width = up(exact.width());
            if width > widest {
                widest = width;
            }
            let magnitude = up(Float::with_val(BOUND_BITS, tau[index].abs_ref()));
            if magnitude > scale {
                scale = magnitude;
            }
            if let Some(stored) = components {
                pole.add(row, column, &distance_upper(w02, &stored.pole[index]));
                archimedean.add(row, column, &distance_upper(wr, &stored.archimedean[index]));
                prime.add(row, column, &distance_upper(wp, &stored.prime[index]));
            }
            Ok(())
        },
    )?;
    let full_norms = full.norms();
    let even_stored = build_even_sector_matrix(tau, n, p)?;
    let even_norms = sector_norms(&bounds, n, &even_stored, p, false)?;
    let odd_norms = if n > 0 {
        Some(sector_norms(
            &bounds,
            n,
            &build_odd_sector_matrix(tau, n, p)?,
            p,
            true,
        )?)
    } else {
        None
    };
    let mode_profile = (0..=n)
        .map(|mode| {
            let rows = [n + mode, n - mode];
            let maximum = rows
                .iter()
                .flat_map(|&row| bounds[row * dimension..(row + 1) * dimension].iter())
                .max_by(|a, b| a.total_cmp(b))
                .cloned()
                .unwrap_or_else(|| Float::with_val(BOUND_BITS, 0));
            decimal_up(&maximum)
        })
        .collect();
    let spectral = parse_bound(&full_norms.spectral_upper)?;
    let surviving_digits = (!spectral.is_zero() && !scale.is_zero()).then(|| {
        let ratio = Float::with_val_round(BOUND_BITS, &spectral / &scale, Round::Up).0;
        Float::with_val(BOUND_BITS, -ratio.log10()).floor().to_f64() as i64
    });
    let even_allowance = parse_bound(&even_norms.spectral_upper)?;
    let odd_allowance = match &odd_norms {
        Some(norms) => Some(parse_bound(&norms.spectral_upper)?),
        None => None,
    };
    let mut exact_form_bounds = Vec::new();
    if let Some(primary) = sources.primary {
        if let Ok(accuracy) = primary.stored_eigenvalue_accuracy() {
            let value = &primary.weil_min_eigenvalue;
            let radius = &accuracy.absolute_error_upper;
            let work = value.prec().max(radius.prec());
            exact_form_bounds.push(exact_form_bound(
                "selected_even_eigenvalue",
                &Float::with_val_round(work, value - radius, Round::Down).0,
                &Float::with_val_round(work, value + radius, Round::Up).0,
                &even_allowance,
            ));
        }
    }
    if let Some(sectors) = sources.sectors {
        for (label, spectrum, allowance) in [
            ("even", &sectors.even, Some(&even_allowance)),
            ("odd", &sectors.odd, odd_allowance.as_ref()),
        ] {
            let Some(allowance) = allowance else {
                continue;
            };
            for pair in &spectrum.eigenpairs {
                exact_form_bounds.push(exact_form_bound(
                    &format!("{label}_eigenvalue_{}", pair.algebraic_index),
                    &pair.eigenvalue_lower,
                    &pair.eigenvalue_upper,
                    allowance,
                ));
            }
            if let [first, second, ..] = spectrum.eigenpairs.as_slice() {
                let work = first.eigenvalue_lower.prec().max(BOUND_BITS);
                let twice = mul_up(allowance, &Float::with_val(BOUND_BITS, 2));
                exact_form_bounds.push(exact_form_bound(
                    &format!("{label}_gap_1_0"),
                    &Float::with_val_round(
                        work,
                        &second.eigenvalue_lower - &first.eigenvalue_upper,
                        Round::Down,
                    )
                    .0,
                    &Float::with_val_round(
                        work,
                        &second.eigenvalue_upper - &first.eigenvalue_lower,
                        Round::Up,
                    )
                    .0,
                    &twice,
                ));
            }
        }
        if let (Some(even), Some(odd), Some(odd_allowance)) = (
            sectors.even.eigenpairs.first(),
            sectors.odd.eigenpairs.first(),
            odd_allowance.as_ref(),
        ) {
            let work = even.eigenvalue_lower.prec().max(BOUND_BITS);
            exact_form_bounds.push(exact_form_bound(
                "odd_minus_even_lowest",
                &Float::with_val_round(
                    work,
                    &odd.eigenvalue_lower - &even.eigenvalue_upper,
                    Round::Down,
                )
                .0,
                &Float::with_val_round(
                    work,
                    &odd.eigenvalue_upper - &even.eigenvalue_lower,
                    Round::Up,
                )
                .0,
                &add_up(&even_allowance, odd_allowance),
            ));
        }
    }
    let second_lower = sources
        .sectors
        .and_then(|sectors| sectors.even.eigenpairs.get(1))
        .map(|pair| pair.eigenvalue_lower.clone());
    let (exact_form_roots, exact_form_roots_limitation) = match (sources.primary, second_lower) {
        (Some(primary), Some(second)) => {
            match exact_form_roots(params, cfg, tau, primary, &even_allowance, &second) {
                Ok(roots) => (Some(roots), None),
                Err(error) => (None, Some(format!("{error:#}"))),
            }
        }
        (None, _) => (None, Some("retained primary state unavailable".to_owned())),
        (_, None) => (
            None,
            Some(
                "a second even-sector eigenvalue enclosure is required for the state separation"
                    .to_owned(),
            ),
        ),
    };
    Ok(CcmAssemblyErrorAnalysis {
        schema_version: 1,
        semantics: ASSEMBLY_ERROR_SEMANTICS.to_owned(),
        claim_scope: ASSEMBLY_ERROR_SCOPE.to_owned(),
        lambda_squared: cutoff,
        prime_cutoff: resolved.prime_cutoff,
        n_modes: n,
        precision_bits: p,
        analysis_precision_bits: analysis_bits,
        geometric_terms: resolved.geometric_terms,
        tau_content_digest: tau_digest.0.clone(),
        matrix_scale: decimal_up(&scale),
        maximum_enclosure_width: decimal_up(&widest),
        full: full_norms,
        even_sector: even_norms,
        odd_sector: odd_norms,
        components: components.map(|_| CcmComponentErrors {
            pole: pole.norms(),
            archimedean: archimedean.norms(),
            prime: prime.norms(),
        }),
        mode_profile,
        surviving_digits,
        exact_form_bounds,
        exact_form_roots,
        exact_form_roots_limitation,
        outcome: "certified_finite_enclosure".to_owned(),
    })
}

/// Enclose the roots of every secular function whose data lie in the exact-form
/// uncertainty box around the retained state, which contains the exact finite
/// form's secular function. The stored and exact forms are both invariant under
/// the reflection n -> -n, so their difference maps even states to even states
/// and the even-sector bound applies to the even retained state.
#[cfg(feature = "arb")]
pub(super) fn exact_form_roots(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    tau: &[Float],
    primary: &HighPrecResult,
    even_allowance: &Float,
    second_lower: &Float,
) -> Result<CcmExactFormRoots> {
    use super::super::certified_roots::CertifiedSecularFunction;
    let p = cfg.precision_bits;
    let q = p.saturating_add(64);
    let xi = &primary.xi;
    let mu = &primary.weil_min_eigenvalue;
    let d = xi.len();
    if d != params.matrix_size() {
        bail!("retained state dimension differs from the source");
    }
    let residual = super::state_residual_bounds::evaluate(tau, xi, mu, p)?.eigenvalue_error_upper;
    let exact_residual = add_up(&up(residual), even_allowance);
    let shifted = Float::with_val_round(q, second_lower - even_allowance, Round::Down).0;
    let separation = Float::with_val_round(q, &shifted - mu, Round::Down).0;
    if separation <= 0 || Float::with_val(q, &exact_residual) >= separation {
        bail!("the retained state is not separated from the rest of the exact even spectrum at assembly scale");
    }
    let sine = Float::with_val_round(BOUND_BITS, &exact_residual / &separation, Round::Up).0;
    if sine >= 1 {
        bail!("eigenvector angle bound is not informative");
    }
    // For unit vectors at angle theta <= pi/2, ||x-u|| = 2 sin(theta/2) <= sqrt(2) sin(theta).
    let kappa = mul_up(&sine, &parse_bound("1.4142135623730952")?);
    let radius = MpfrInterval::new(Float::with_val(q, -&kappa), Float::with_val(q, &kappa))?;
    let entries = xi
        .iter()
        .map(|value| MpfrInterval::from_float(value, q))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mut squares = MpfrInterval::from_i64(0, q);
    let mut sum = MpfrInterval::from_i64(0, q);
    for entry in &entries {
        squares = squares.add(&entry.square());
        sum = sum.add(entry);
    }
    let norm = squares.sqrt()?;
    let alignment = sum.div(&norm)?;
    let spread = MpfrInterval::from_u64(d as u64, q).sqrt()?.mul(&radius);
    let denominator = alignment.add(&spread);
    if !denominator.is_strictly_positive() {
        bail!("the normalization sum is not resolved at the eigenvector uncertainty");
    }
    let length = super::super::retained_evidence::finite_math::decimal(
        &lambda_squared_cache_identity(params),
        q,
    )?
    .ln()?;
    let scale = length.sqrt()?;
    let weights = entries
        .iter()
        .map(|entry| {
            Ok(scale
                .mul(&entry.div(&norm)?.add(&radius))
                .div(&denominator)?)
        })
        .collect::<Result<Vec<_>>>()?;
    let two_pi = MpfrInterval::pi(q).mul(&MpfrInterval::from_i64(2, q));
    let modes = params.n_modes as i64;
    let poles = (-modes..=modes)
        .map(|j| two_pi.mul(&MpfrInterval::from_i64(j, q)).div(&length))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let source = CertifiedSecularFunction::from_interval_data(poles, weights)?;
    let rows = primary
        .eigenvalues_pos
        .iter()
        .enumerate()
        .filter_map(|(offset, outcome)| {
            outcome.value().map(|value| {
                enclose_root(
                    &source,
                    value,
                    primary.first_positive_root_index + offset,
                    q,
                )
            })
        })
        .collect();
    Ok(CcmExactFormRoots {
        state_residual_upper: decimal_up(&exact_residual),
        separation_lower: decimal_down(&separation),
        eigenvector_sine_upper: decimal_up(&sine),
        vector_distance_upper: decimal_up(&kappa),
        rows,
    })
}

#[cfg(feature = "arb")]
fn enclose_root(
    source: &super::super::certified_roots::CertifiedSecularFunction,
    value: &Float,
    ordinal: usize,
    q: u32,
) -> CcmExactFormRoot {
    use xc_root::RealIntervalFunctionHp;
    let mut row = CcmExactFormRoot {
        ordinal,
        computed_value: lossless_hp_decimal(value),
        outcome: "unresolved".to_owned(),
        exact_form_lower: None,
        exact_form_upper: None,
        exact_form_resolved_digits: None,
        reason: None,
    };
    let attempt = || -> Result<Option<(String, String)>> {
        let center = MpfrInterval::from_float(value, q)?;
        let residual = source.evaluate_interval(&center)?;
        let slope = source.derivative_interval(&center)?;
        if slope.contains_zero() {
            bail!("the secular derivative is not resolved at the computed root");
        }
        let magnitude = Float::with_val(q, residual.lower().abs_ref())
            .max(&Float::with_val(q, residual.upper().abs_ref()));
        let mignitude = Float::with_val(q, slope.lower().abs_ref())
            .min(&Float::with_val(q, slope.upper().abs_ref()));
        let floor = Float::with_val(q, value.abs_ref()).max(&Float::with_val(q, 1)) >> (q - 16);
        let mut radius = Float::with_val_round(q, &magnitude * 4u32, Round::Up).0;
        radius = Float::with_val_round(q, &radius / &mignitude, Round::Up).0;
        radius = radius.max(&floor);
        for _ in 0..4 {
            let lower = Float::with_val_round(q, value - &radius, Round::Down).0;
            let upper = Float::with_val_round(q, value + &radius, Round::Up).0;
            let options = xc_root::IntervalNewtonOptions {
                width_tolerance: xc_core::DecimalLiteral::new(radius.to_string_radix_round(
                    10,
                    Some(20),
                    Round::Down,
                ))?,
                maximum_iterations: 64,
            };
            if let Ok(certificate) = source.isolate(&MpfrInterval::new(lower, upper)?, &options) {
                if certificate.uniqueness_witnessed {
                    return Ok(Some((certificate.lower, certificate.upper)));
                }
            }
            radius *= 16u32;
        }
        Ok(None)
    };
    match attempt() {
        Ok(Some((lower, upper))) => {
            row.exact_form_resolved_digits = resolved_digits(&lower, &upper, q);
            row.exact_form_lower = Some(lower);
            row.exact_form_upper = Some(upper);
            row.outcome = "enclosed".to_owned();
        }
        Ok(None) => {
            row.reason = Some(
                "interval Newton did not establish a unique root at the assembly scale".to_owned(),
            )
        }
        Err(error) => row.reason = Some(format!("{error:#}")),
    }
    row
}

/// Resolve or compute the analysis through the managed cache. The stored
/// components are rebuilt from the same deterministic point stages as Tau,
/// reusing cached quadrature and integral dependencies.
#[cfg(feature = "arb")]
pub(super) fn resolve_assembly_error_via_cache(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    tau: &[Float],
    tau_manifest: &ArtifactManifest,
    primary: Option<(&HighPrecResult, &ArtifactManifest)>,
    sectors: Option<(&CcmSectorGapHp, &ArtifactManifest)>,
    cache: &ArtifactCacheContext<'_>,
) -> Result<xc_cache::ArtifactExecutionCacheResult<CcmAssemblyErrorAnalysis>> {
    let parity_policy = cfg.effective_parity_policy();
    let mut resolved_parameters = serde_json::json!({
        "lambda_squared": lambda_squared_cache_identity(params),
        "n_modes": params.n_modes,
        "precision_bits": cfg.precision_bits,
        "force_even": parity_policy.legacy_force_even(),
        "tau_content_digest": tau_manifest.content_digest.0,
        "eigenpair_content_digest": primary.map(|(_, m)| m.content_digest.0.clone()),
        "sector_gap_content_digest": sectors.map(|(_, m)| m.content_digest.0.clone()),
        "analysis_guard_bits": 64,
    });
    add_adaptive_parity_parameter(&mut resolved_parameters, parity_policy);
    let mut identities = BTreeMap::from([(
        "ccm_tau_matrix".to_owned(),
        tau_manifest.content_digest.clone(),
    )]);
    let mut dependencies = vec![tau_manifest.clone()];
    if let Some((_, manifest)) = primary {
        identities.insert(
            "ccm_weil_eigenpair".to_owned(),
            manifest.content_digest.clone(),
        );
        dependencies.push(manifest.clone());
    }
    if let Some((_, manifest)) = sectors {
        identities.insert("ccm_sector_gap".to_owned(), manifest.content_digest.clone());
        dependencies.push(manifest.clone());
    }
    let semantic_key = SemanticKeyEnvelope {
        schema_version: 1,
        artifact_kind: "ccm_assembly_error_analysis".to_owned(),
        mathematical_semantics_version: ASSEMBLY_ERROR_SEMANTICS.to_owned(),
        resolved_mathematical_parameters: resolved_parameters,
        normalization: None,
        target: Some("stored_minus_exact_finite_weil_form".to_owned()),
        subspace: parity_policy.semantic_subspace(),
        source_data_identities: identities,
        algorithm_semantics: Some(
            "cutoff_free_arb_streamed_entry_bounds_weyl_parity_rounding_v1".to_owned(),
        ),
    };
    let semantic_digest = semantic_key.digest()?;
    let logical_key = format!(
        "ccm/assembly-error/{}/{}/{}/{}",
        lambda_squared_cache_identity(params),
        params.n_modes,
        cfg.precision_bits,
        semantic_digest.0
    );
    let request = ArtifactExecutionCacheRequest {
        operation: "ccm.assembly_error_analysis.resolve_or_compute",
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
            ("artifact".to_owned(), "assembly_error_analysis".to_owned()),
        ]),
        provenance_digest: Some(tau_manifest.content_digest.clone()),
        production_sink: cache.production_sink,
    };
    let validate = |analysis: &CcmAssemblyErrorAnalysis| {
        if analysis.schema_version != 1
            || analysis.semantics != ASSEMBLY_ERROR_SEMANTICS
            || analysis.claim_scope != ASSEMBLY_ERROR_SCOPE
            || analysis.lambda_squared != lambda_squared_cache_identity(params)
            || analysis.n_modes != params.n_modes
            || analysis.precision_bits != cfg.precision_bits
            || analysis.tau_content_digest != tau_manifest.content_digest.0
            || analysis.mode_profile.len() != params.n_modes + 1
            || analysis.outcome != "certified_finite_enclosure"
        {
            return Err(CacheError::InvalidManifest(
                "CCM assembly error analysis does not match its semantic identity".to_owned(),
            ));
        }
        Ok(())
    };
    Ok(resolve_or_compute_json_artifact_with_dependencies(
        &request,
        || {
            let l = log_lambda_sq_hp(params, cfg.precision_bits)
                .map_err(|error| CacheError::InvalidManifest(error.to_string()))?;
            let (components, _) = build_tau_components_exact_tracked(
                params.n_modes,
                params.lambda_sq_int(),
                &l,
                cfg,
                true,
                Some(cache),
            )
            .map_err(|error| CacheError::InvalidManifest(format!("{error:#}")))?;
            let analysis = analyze_assembly_error(
                params,
                cfg,
                tau,
                &tau_manifest.content_digest,
                Some(&components),
                ExactFormSources {
                    primary: primary.map(|(result, _)| result),
                    sectors: sectors.map(|(gap, _)| gap),
                },
            )
            .map_err(|error| {
                CacheError::InvalidManifest(format!("CCM assembly error: {error:#}"))
            })?;
            Ok((analysis, canonical_dependency_refs(dependencies)))
        },
        validate,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assembly_error_is_small_and_detects_perturbations_for_integer_and_fractional_cutoffs() {
        for params in [
            CcmParams::from_lambda_sq_integer(13, 4),
            CcmParams::from_lambda_sq_fractional(12.5, 4),
        ] {
            let mut cfg = HighPrecConfig::for_decimal_digits(40);
            cfg.precision_bits = 192;
            cfg.quad_points = MIN_QUAD_POINTS;
            cfg.cache_mode = xc_numerics::quadrature::CacheMode::Off;
            let tau = weil_matrix_hp(&params, &cfg, true).unwrap();
            let l = log_lambda_sq_hp(&params, cfg.precision_bits).unwrap();
            let (components, _) = build_tau_components_exact_tracked(
                params.n_modes,
                params.lambda_sq_int(),
                &l,
                &cfg,
                true,
                None,
            )
            .unwrap();
            let digest = ContentDigest::sha256(b"tau");
            let analyze = |tau: &[Float]| {
                analyze_assembly_error(
                    &params,
                    &cfg,
                    tau,
                    &digest,
                    Some(&components),
                    ExactFormSources {
                        primary: None,
                        sectors: None,
                    },
                )
                .unwrap()
            };
            let analysis = analyze(&tau);
            let spectral = parse_bound(&analysis.full.spectral_upper).unwrap();
            let scale = parse_bound(&analysis.matrix_scale).unwrap();
            assert!(spectral < Float::with_val(BOUND_BITS, &scale >> (cfg.precision_bits - 40)));
            assert!(analysis.surviving_digits.unwrap() >= 40, "{analysis:?}");
            assert_eq!(analysis.mode_profile.len(), params.n_modes + 1);
            assert!(analysis.components.is_some());
            assert!(parse_bound(&analysis.even_sector.spectral_upper).unwrap() < scale);

            assert!(analysis.exact_form_roots.is_none());
            assert!(analysis.exact_form_roots_limitation.is_some());
            let d = params.matrix_size();
            let delta = Float::with_val(cfg.precision_bits, 1) >> 30u32;
            let mut perturbed = tau.clone();
            perturbed[1] += &delta;
            perturbed[d] += &delta;
            let detected = analyze(&perturbed);
            let half = Float::with_val(BOUND_BITS, &delta / 2u32);
            assert!(parse_bound(&detected.full.max_entry_upper).unwrap() >= half);
            assert!(parse_bound(&detected.full.spectral_upper).unwrap() >= half);
        }
    }
}

#[cfg(all(test, feature = "arb"))]
mod exact_form_root_tests {
    use super::super::root_certification_report::build_root_certification_report;
    use super::*;

    #[test]
    fn exact_form_root_enclosures_agree_with_certified_stored_roots() {
        let params = CcmParams::from_lambda_sq_integer(13, 10);
        let mut cfg = HighPrecConfig::for_decimal_digits(40);
        cfg.precision_bits = 192;
        cfg.quad_points = MIN_QUAD_POINTS;
        cfg.cache_mode = xc_numerics::quadrature::CacheMode::Off;
        let p = cfg.precision_bits;
        let tau = weil_matrix_hp(&params, &cfg, true).unwrap();
        let (mut primary, _) = run_inner_retaining_source(
            &params,
            &cfg,
            RootAcquisition::SourceOnly,
            CcmCacheRoute::Standalone,
            None,
        )
        .unwrap();
        primary.first_positive_root_index = 1;
        primary.eigenvalues_pos.clear();
        let options = CcmRootCertificationOptions::for_decimal_digits(
            super::super::super::certified_roots::IndependentCcmRootTarget::Prefix { count: 2 },
            30,
        )
        .unwrap();
        let digest = ContentDigest::sha256(b"source");
        let report = build_root_certification_report(
            &params,
            &cfg,
            &primary,
            &primary.xi,
            &options,
            &digest,
            &digest,
        )
        .unwrap();
        assert_eq!(report.certified_rows, 2);
        let midpoints = report
            .rows
            .iter()
            .map(|row| {
                let lower = Float::with_val(
                    p,
                    Float::parse(row.certified_lower.as_ref().unwrap()).unwrap(),
                );
                let upper = Float::with_val(
                    p,
                    Float::parse(row.certified_upper.as_ref().unwrap()).unwrap(),
                );
                Float::with_val(p, lower + upper) / 2u32
            })
            .collect::<Vec<_>>();
        primary.eigenvalues_pos = midpoints
            .iter()
            .map(|value| {
                EigenvalueResult::Converged(RootRefinement {
                    value: value.clone(),
                    diagnostics: RootRefinementDiagnostics {
                        iterations: 1,
                        final_correction: Float::with_val(p, 0),
                        residual: Float::with_val(p, 0),
                        achieved_decimal_digits: Float::with_val(p, 30),
                    },
                })
            })
            .collect();
        let even = build_even_sector_matrix(&tau, params.n_modes, p).unwrap();
        let spectrum =
            super::super::spectrum_accuracy::lowest_spectrum(&even, params.n_modes + 1, p, 2)
                .unwrap();
        let analysis = analyze_assembly_error(
            &params,
            &cfg,
            &tau,
            &digest,
            None,
            ExactFormSources {
                primary: None,
                sectors: None,
            },
        )
        .unwrap();
        let allowance = parse_bound(&analysis.even_sector.spectral_upper).unwrap();
        let roots = exact_form_roots(
            &params,
            &cfg,
            &tau,
            &primary,
            &allowance,
            &spectrum.eigenvalue_bounds[1].lower,
        )
        .unwrap();
        assert_eq!(roots.rows.len(), 2);
        for (row, midpoint) in roots.rows.iter().zip(&midpoints) {
            assert_eq!(row.outcome, "enclosed", "{row:?}");
            let digits: f64 = row
                .exact_form_resolved_digits
                .as_ref()
                .unwrap()
                .parse()
                .unwrap();
            assert!(digits >= 15.0, "{row:?}");
            let q = p + 64;
            let lower = Float::with_val(
                q,
                Float::parse(row.exact_form_lower.as_ref().unwrap()).unwrap(),
            );
            let upper = Float::with_val(
                q,
                Float::parse(row.exact_form_upper.as_ref().unwrap()).unwrap(),
            );
            let slack = Float::with_val(q, &upper - &lower) + (Float::with_val(q, 1) >> (p - 20));
            assert!(Float::with_val(q, midpoint - &lower) >= -slack.clone());
            assert!(Float::with_val(q, &upper - midpoint) >= -slack);
        }
    }
}
