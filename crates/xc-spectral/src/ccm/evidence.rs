// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Rigorous finite-claim evidence for CCM and prolate comparisons.
//!
//! This module deliberately keeps three quantities separate:
//!
//! - a finite prolate deficiency derived from a certified concentration
//!   eigenvalue enclosure;
//! - the asymptotic predictor for `1 - chi_2(lambda)`;
//! - the measured finite Weil plunge.
//!
//! It also implements the explicit archimedean tail budget and decision rule
//! of Groskin, Theorem 3.2 and Corollary 3.3 (arXiv:2607.02828).  The budget is
//! never constructed outside its theorem domain.

use anyhow::{bail, Result};
use rug::float::Constant;
use rug::{Float, Rational};
use xc_numerics::interval::RationalInterval;

/// An explicit operator-norm upper bound for the omitted archimedean tail.
#[derive(Clone, Debug)]
pub struct ArchimedeanTailBudget {
    /// `L = log(c)`.
    pub log_cutoff: Float,
    /// `rho = 2*pi/L`.
    pub rho: Float,
    /// Finite frequency band `{-N, ..., N}`.
    pub modes: usize,
    /// Archimedean integration cutoff `T`.
    pub cutoff_t: Float,
    /// The strict theorem threshold `max(rho*N, 7)`.
    pub theorem_threshold: Float,
    /// Explicit upper bound `B_T` from Corollary 3.3(iii).
    pub upper_bound: Float,
}

impl ArchimedeanTailBudget {
    /// Evaluate the published explicit tail bound.
    ///
    /// The theorem requires integer `c > 1`, `N >= 1`, and
    /// `T > max(rho*N, 7)`.  Invalid requests fail rather than extrapolate the
    /// formula beyond its hypotheses.
    pub fn explicit(integer_cutoff_c: u64, modes: usize, cutoff_t: &Float) -> Result<Self> {
        if integer_cutoff_c <= 1 {
            bail!("archimedean tail budget requires integer cutoff c > 1");
        }
        if modes == 0 {
            bail!("the explicit Corollary 3.3 budget requires N >= 1");
        }
        let precision = cutoff_t.prec();
        if !cutoff_t.is_finite() || cutoff_t <= &Float::with_val(precision, 0) {
            bail!("archimedean cutoff T must be finite and positive");
        }

        let log_cutoff = Float::with_val(precision, integer_cutoff_c).ln();
        let pi = Float::with_val(precision, Constant::Pi);
        let mut rho = pi.clone();
        rho *= 2u32;
        rho /= &log_cutoff;
        let mut band_edge = rho.clone();
        band_edge *= modes as u32;
        let seven = Float::with_val(precision, 7);
        let theorem_threshold = if band_edge > seven { band_edge } else { seven };
        if cutoff_t <= &theorem_threshold {
            bail!("archimedean cutoff T must be strictly greater than max(rho*N, 7)");
        }

        // B_T <= 2(2N+1)rho/pi^2 *
        //   [log(T)/(T-rho*N) + log(T/(T-rho*N))/(rho*N)].
        let mut denominator = cutoff_t.clone();
        let mut rho_n = rho.clone();
        rho_n *= modes as u32;
        denominator -= &rho_n;

        let mut first = cutoff_t.clone().ln();
        first /= &denominator;
        let mut ratio = cutoff_t.clone();
        ratio /= &denominator;
        let mut second = ratio.ln();
        second /= &rho_n;
        first += second;

        let mut prefactor = rho.clone();
        prefactor *= 2u32;
        prefactor *= (2 * modes + 1) as u32;
        let mut pi_squared = pi;
        pi_squared.square_mut();
        prefactor /= pi_squared;
        let mut upper_bound = prefactor;
        upper_bound *= first;

        Ok(Self {
            log_cutoff,
            rho,
            modes,
            cutoff_t: cutoff_t.clone(),
            theorem_threshold,
            upper_bound,
        })
    }
}

/// What a finite-`T` eigenvalue and a valid tail budget prove about the
/// corresponding cutoff-free eigenvalue.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FiniteCutoffDecision {
    /// `lambda_j(Q_T) >= 0`, hence `lambda_j(Q_infinity) > 0`.
    CutoffFreePositive,
    /// `lambda_j(Q_T) < -B_T`, hence `lambda_j(Q_infinity) < 0`.
    CutoffFreeNegative,
    /// The finite value lies in `[-B_T, 0)` and has no cutoff-free sign.
    InconclusiveTailBand,
}

pub fn finite_cutoff_decision(
    finite_t_eigenvalue: &Float,
    budget: &ArchimedeanTailBudget,
) -> FiniteCutoffDecision {
    let zero = Float::with_val(finite_t_eigenvalue.prec(), 0);
    if finite_t_eigenvalue >= &zero {
        return FiniteCutoffDecision::CutoffFreePositive;
    }
    let mut negative_budget = budget.upper_bound.clone();
    negative_budget = -negative_budget;
    if finite_t_eigenvalue < &negative_budget {
        FiniteCutoffDecision::CutoffFreeNegative
    } else {
        FiniteCutoffDecision::InconclusiveTailBand
    }
}

/// Certified finite evaluation of the positive prolate angular deficiency
/// `1 - chi_2(lambda)` associated with `h_{4,lambda}`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CertifiedProlateDeficiency {
    /// Certified enclosure of `nu_2 = chi_2^2`.
    pub concentration_eigenvalue: RationalInterval,
    /// Certified enclosure of the positive square root `chi_2`.
    pub angular_eigenvalue: RationalInterval,
    /// Certified enclosure of `1 - chi_2`.
    pub deficiency: RationalInterval,
    /// Whether the finite deficiency was represented exactly by the dyadic
    /// square-root grid rather than by a nonzero-width enclosure.
    pub exact_finite: bool,
    pub fraction_bits: u32,
}

impl CertifiedProlateDeficiency {
    /// Propagate a certified enclosure of `nu_2` through the positive square
    /// root using exact rational, outward-rounded arithmetic.
    pub fn from_concentration_enclosure(
        concentration_eigenvalue: RationalInterval,
        fraction_bits: u32,
    ) -> Result<Self> {
        let zero = Rational::from((0, 1));
        let one = Rational::from((1, 1));
        if concentration_eigenvalue.lower() < &zero || concentration_eigenvalue.upper() > &one {
            bail!("certified prolate concentration eigenvalue must lie in [0,1]");
        }
        let angular_eigenvalue = concentration_eigenvalue
            .sqrt_nonnegative(fraction_bits)
            .map_err(anyhow::Error::new)?;
        let deficiency = RationalInterval::point(one).sub(&angular_eigenvalue);
        let exact_finite = deficiency.is_point();
        Ok(Self {
            concentration_eigenvalue,
            angular_eigenvalue,
            deficiency,
            exact_finite,
            fraction_bits,
        })
    }
}

/// End-to-end finite prolate evidence: an exact shifted-inertia certificate
/// for the selected concentration eigenvalue and its outward-rounded angular
/// deficiency propagation.
#[derive(Clone, Debug)]
pub struct CertifiedProlateDeficiencyEvidence {
    pub selected_eigenvalue: xc_certify::ExactSelectedEigenvalueEnclosure,
    pub deficiency: CertifiedProlateDeficiency,
}

#[derive(Clone, Debug)]
pub struct ProlateConcentrationCertificationRequest {
    pub dimension: usize,
    pub requested_index: usize,
    pub lower_bracket: Rational,
    pub upper_bracket: Rational,
    pub target_width: Rational,
    pub maximum_bisection_steps: usize,
    pub square_root_fraction_bits: u32,
}

/// Generate the selected concentration-eigenvalue enclosure rather than
/// accepting an unaudited interval from the caller, verify it against the
/// exact interval matrix, and propagate it through `1 - sqrt(nu)`.
pub fn certify_prolate_deficiency_from_concentration_matrix(
    concentration_matrix: &[RationalInterval],
    request: ProlateConcentrationCertificationRequest,
) -> Result<CertifiedProlateDeficiencyEvidence> {
    let result = xc_certify::exact::certify_selected_interval_eigenvalue(
        concentration_matrix,
        request.dimension,
        request.requested_index,
        request.lower_bracket,
        request.upper_bracket,
        request.target_width,
        request.maximum_bisection_steps,
    );
    let certificate = match result {
        xc_certify::SelectedEigenvalueEnclosureResult::Conclusive { certificate } => certificate,
        xc_certify::SelectedEigenvalueEnclosureResult::Inconclusive { boundary, reason } => {
            bail!("prolate concentration eigenvalue was inconclusive at {boundary}: {reason}")
        }
    };
    if !certificate.simple {
        bail!("the selected prolate concentration eigenvalue is not certified simple");
    }
    let replay = xc_certify::exact::verify_selected_interval_eigenvalue_enclosure(
        &certificate,
        concentration_matrix,
    );
    if !replay.valid {
        bail!(
            "selected prolate concentration certificate failed exact replay: {}",
            replay.errors.join("; ")
        );
    }
    let enclosure = RationalInterval::new(
        xc_certify::exact::parse(&certificate.lower)?,
        xc_certify::exact::parse(&certificate.upper)?,
    )?;
    let deficiency = CertifiedProlateDeficiency::from_concentration_enclosure(
        enclosure,
        request.square_root_fraction_bits,
    )?;
    Ok(CertifiedProlateDeficiencyEvidence {
        selected_eigenvalue: *certificate,
        deficiency,
    })
}

/// Keep the finite prolate, asymptotic, and measured Weil quantities in one
/// comparison without allowing any of them to serve as the other's oracle.
#[derive(Clone, Debug)]
pub struct ProlateWeilComparison {
    pub finite_prolate: CertifiedProlateDeficiency,
    pub asymptotic_predictor: Float,
    pub measured_weil_plunge: RationalInterval,
    /// `measured_weil_plunge - finite_prolate.deficiency`.
    pub finite_difference: RationalInterval,
}

impl ProlateWeilComparison {
    pub fn new(
        lambda_squared: u64,
        finite_prolate: CertifiedProlateDeficiency,
        measured_weil_plunge: RationalInterval,
        precision_bits: u32,
    ) -> Result<Self> {
        if lambda_squared <= 1 {
            bail!("prolate asymptotic predictor requires lambda^2 > 1");
        }
        let asymptotic_predictor =
            prolate_chi2_deficiency_asymptotic(lambda_squared, precision_bits);
        let finite_difference = measured_weil_plunge.sub(&finite_prolate.deficiency);
        Ok(Self {
            finite_prolate,
            asymptotic_predictor,
            measured_weil_plunge,
            finite_difference,
        })
    }
}

/// The large-`lambda` predictor
/// `(2^14/3)*sqrt(2*pi)*exp(-4*pi*exp(L) + 9L/2)`, `L=log(lambda^2)`.
///
/// This is intentionally labeled as an asymptotic predictor, never a finite
/// certificate.
pub fn prolate_chi2_deficiency_asymptotic(lambda_squared: u64, precision_bits: u32) -> Float {
    let pi = Float::with_val(precision_bits, Constant::Pi);
    let l = Float::with_val(precision_bits, lambda_squared).ln();
    let mut exponent = pi.clone();
    exponent *= lambda_squared;
    exponent *= -4i32;
    let mut nine_l_over_two = l;
    nine_l_over_two *= 9u32;
    nine_l_over_two /= 2u32;
    exponent += nine_l_over_two;

    let mut prefactor = pi;
    prefactor *= 2u32;
    prefactor.sqrt_mut();
    prefactor *= 2u32.pow(14);
    prefactor /= 3u32;
    prefactor * exponent.exp()
}

/// A rigorous residual-to-gap conclusion for a normalized approximate
/// eigenvector.  With residual `r` and certified gap `delta`, the standard
/// simple-eigenvector bound gives `sin(angle) <= r/delta`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResidualGapConclusion {
    Reliable {
        residual_upper: Rational,
        gap_lower: Rational,
        sin_angle_upper: Rational,
    },
    Inconclusive {
        residual_upper: Rational,
        gap_lower: Rational,
    },
}

pub fn residual_to_certified_gap(
    residual_upper: Rational,
    gap_lower: Rational,
) -> Result<ResidualGapConclusion> {
    if residual_upper < 0 {
        bail!("residual upper bound must be non-negative");
    }
    if gap_lower <= 0 {
        bail!("certified spectral-gap lower bound must be positive");
    }
    if residual_upper >= gap_lower {
        return Ok(ResidualGapConclusion::Inconclusive {
            residual_upper,
            gap_lower,
        });
    }
    let mut sin_angle_upper = residual_upper.clone();
    sin_angle_upper /= &gap_lower;
    Ok(ResidualGapConclusion::Reliable {
        residual_upper,
        gap_lower,
        sin_angle_upper,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComparisonTruncationKind {
    ValueSpace,
    CoefficientSpace,
    FormNorm,
}

#[derive(Clone, Debug)]
pub struct ActiveTruncationBound {
    pub kind: ComparisonTruncationKind,
    pub upper_bound: Float,
    pub source: String,
}

/// Direct finite-dimensional comparison of Connes's prolate candidate and a
/// computed Weil state. Both inputs are normalized internally, and the
/// prolate sign is aligned to the Weil state before differences are formed.
#[derive(Clone, Debug)]
pub struct ProlateWeilStateComparisonHp {
    pub value_space_overlap: Float,
    pub value_space_residual: Float,
    pub coefficient_overlap: Float,
    pub coefficient_residual: Float,
    pub prolate_rayleigh_quotient: Float,
    pub weil_rayleigh_quotient: Float,
    pub prolate_eigen_residual: Float,
    pub weil_eigen_residual: Float,
    pub form_norm_difference: Float,
    pub truncation_bounds: Vec<ActiveTruncationBound>,
}

fn hp_dot(left: &[Float], right: &[Float], precision_bits: u32) -> Result<Float> {
    if left.len() != right.len() || left.is_empty() {
        bail!("HP comparison vectors must have the same nonzero dimension");
    }
    let terms = left
        .iter()
        .zip(right)
        .map(|(left, right)| {
            let mut term = Float::with_val(precision_bits, left);
            term *= right;
            term
        })
        .collect::<Vec<_>>();
    Ok(xc_numerics::reduction::deterministic_pairwise_sum_hp(
        &terms,
        precision_bits,
    ))
}

fn hp_normalize(vector: &[Float], precision_bits: u32) -> Result<Vec<Float>> {
    let norm_squared = hp_dot(vector, vector, precision_bits)?;
    if !norm_squared.is_finite() || norm_squared <= 0 {
        bail!("HP comparison vectors must have finite positive norm");
    }
    let norm = norm_squared.sqrt();
    Ok(vector
        .iter()
        .map(|value| {
            let mut normalized = Float::with_val(precision_bits, value);
            normalized /= &norm;
            normalized
        })
        .collect())
}

fn hp_matvec(matrix: &[Float], vector: &[Float], precision_bits: u32) -> Result<Vec<Float>> {
    let dimension = vector.len();
    if dimension == 0 || matrix.len() != dimension * dimension {
        bail!("HP comparison form must be a nonempty square matrix");
    }
    Ok((0..dimension)
        .map(|row| {
            let terms = (0..dimension)
                .map(|column| {
                    let mut term =
                        Float::with_val(precision_bits, &matrix[row * dimension + column]);
                    term *= &vector[column];
                    term
                })
                .collect::<Vec<_>>();
            xc_numerics::reduction::deterministic_pairwise_sum_hp(&terms, precision_bits)
        })
        .collect())
}

fn hp_l2_norm(vector: &[Float], precision_bits: u32) -> Result<Float> {
    Ok(hp_dot(vector, vector, precision_bits)?.sqrt())
}

fn aligned_difference(
    reference: &[Float],
    candidate: &[Float],
    precision_bits: u32,
) -> Result<(Float, Vec<Float>)> {
    let signed_overlap = hp_dot(reference, candidate, precision_bits)?;
    let sign = if signed_overlap < 0 { -1i32 } else { 1i32 };
    let difference = reference
        .iter()
        .zip(candidate)
        .map(|(reference, candidate)| {
            let mut value = Float::with_val(precision_bits, candidate);
            value *= sign;
            value -= reference;
            value
        })
        .collect();
    Ok((signed_overlap.abs(), difference))
}

/// Compare the two states in sampled value space, coefficient space, and the
/// supplied positive-form norm. Exactly one active bound for each space is
/// mandatory so a report cannot omit the truncation regime under comparison.
pub fn compare_prolate_weil_states_hp(
    weil_values: &[Float],
    prolate_values: &[Float],
    weil_coefficients: &[Float],
    prolate_coefficients: &[Float],
    weil_form: &[Float],
    truncation_bounds: Vec<ActiveTruncationBound>,
    precision_bits: u32,
) -> Result<ProlateWeilStateComparisonHp> {
    for kind in [
        ComparisonTruncationKind::ValueSpace,
        ComparisonTruncationKind::CoefficientSpace,
        ComparisonTruncationKind::FormNorm,
    ] {
        let matches = truncation_bounds
            .iter()
            .filter(|bound| bound.kind == kind)
            .count();
        if matches != 1 {
            bail!("each prolate/Weil comparison space requires exactly one truncation bound");
        }
    }
    if truncation_bounds.iter().any(|bound| {
        !bound.upper_bound.is_finite() || bound.upper_bound < 0 || bound.source.trim().is_empty()
    }) {
        bail!("active truncation bounds require finite nonnegative values and a source");
    }

    let weil_values = hp_normalize(weil_values, precision_bits)?;
    let prolate_values = hp_normalize(prolate_values, precision_bits)?;
    let (value_space_overlap, value_difference) =
        aligned_difference(&weil_values, &prolate_values, precision_bits)?;
    let value_space_residual = hp_l2_norm(&value_difference, precision_bits)?;

    let weil_coefficients = hp_normalize(weil_coefficients, precision_bits)?;
    let prolate_coefficients = hp_normalize(prolate_coefficients, precision_bits)?;
    let (coefficient_overlap, coefficient_difference) =
        aligned_difference(&weil_coefficients, &prolate_coefficients, precision_bits)?;
    let coefficient_residual = hp_l2_norm(&coefficient_difference, precision_bits)?;

    let weil_action = hp_matvec(weil_form, &weil_coefficients, precision_bits)?;
    let prolate_action = hp_matvec(weil_form, &prolate_coefficients, precision_bits)?;
    let weil_rayleigh_quotient = hp_dot(&weil_coefficients, &weil_action, precision_bits)?;
    let prolate_rayleigh_quotient = hp_dot(&prolate_coefficients, &prolate_action, precision_bits)?;
    let residual = |state: &[Float], action: &[Float], rayleigh: &Float| {
        let values = state
            .iter()
            .zip(action)
            .map(|(state, action)| {
                let mut value = Float::with_val(precision_bits, state);
                value *= rayleigh;
                let mut residual = Float::with_val(precision_bits, action);
                residual -= value;
                residual
            })
            .collect::<Vec<_>>();
        hp_l2_norm(&values, precision_bits)
    };
    let weil_eigen_residual = residual(&weil_coefficients, &weil_action, &weil_rayleigh_quotient)?;
    let prolate_eigen_residual = residual(
        &prolate_coefficients,
        &prolate_action,
        &prolate_rayleigh_quotient,
    )?;
    let difference_action = hp_matvec(weil_form, &coefficient_difference, precision_bits)?;
    let form_norm_squared = hp_dot(&coefficient_difference, &difference_action, precision_bits)?;
    if form_norm_squared < 0 {
        bail!("the supplied Weil form is negative on the candidate difference");
    }
    let form_norm_difference = form_norm_squared.sqrt();

    Ok(ProlateWeilStateComparisonHp {
        value_space_overlap,
        value_space_residual,
        coefficient_overlap,
        coefficient_residual,
        prolate_rayleigh_quotient,
        weil_rayleigh_quotient,
        prolate_eigen_residual,
        weil_eigen_residual,
        form_norm_difference,
        truncation_bounds,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(numerator: i32, denominator: i32) -> Rational {
        Rational::from((numerator, denominator))
    }

    #[test]
    fn tail_budget_enforces_theorem_domain_and_decision_band() {
        let precision = 192;
        let invalid_t = Float::with_val(precision, 7);
        assert!(ArchimedeanTailBudget::explicit(13, 4, &invalid_t).is_err());

        let t = Float::with_val(precision, 100);
        let budget = ArchimedeanTailBudget::explicit(13, 4, &t).unwrap();
        assert!(budget.upper_bound > 0);
        assert_eq!(
            finite_cutoff_decision(&Float::with_val(precision, 1), &budget),
            FiniteCutoffDecision::CutoffFreePositive
        );
        let mut deep_negative = budget.upper_bound.clone();
        deep_negative *= -2i32;
        assert_eq!(
            finite_cutoff_decision(&deep_negative, &budget),
            FiniteCutoffDecision::CutoffFreeNegative
        );
        let mut shallow_negative = budget.upper_bound.clone();
        shallow_negative /= -2i32;
        assert_eq!(
            finite_cutoff_decision(&shallow_negative, &budget),
            FiniteCutoffDecision::InconclusiveTailBand
        );
    }

    #[test]
    fn exact_dyadic_prolate_deficiency_stays_exact_and_separate() {
        let nu = RationalInterval::point(q(1, 4));
        let finite = CertifiedProlateDeficiency::from_concentration_enclosure(nu, 64).unwrap();
        assert!(finite.exact_finite);
        assert_eq!(finite.angular_eigenvalue.lower(), &q(1, 2));
        assert_eq!(finite.deficiency.lower(), &q(1, 2));

        let comparison = ProlateWeilComparison::new(
            13,
            finite,
            RationalInterval::new(q(49, 100), q(51, 100)).unwrap(),
            192,
        )
        .unwrap();
        assert!(comparison.finite_difference.contains(&q(0, 1)));
        assert!(comparison.asymptotic_predictor > 0);
    }

    #[test]
    fn concentration_matrix_generates_replayable_deficiency_enclosure() {
        let point = |numerator, denominator| RationalInterval::point(q(numerator, denominator));
        let matrix = vec![point(1, 4), point(0, 1), point(0, 1), point(7, 13)];
        let evidence = certify_prolate_deficiency_from_concentration_matrix(
            &matrix,
            ProlateConcentrationCertificationRequest {
                dimension: 2,
                requested_index: 1,
                lower_bracket: q(1, 2),
                upper_bracket: q(3, 5),
                target_width: q(1, 10_000),
                maximum_bisection_steps: 32,
                square_root_fraction_bits: 96,
            },
        )
        .unwrap();
        assert!(evidence.selected_eigenvalue.simple);
        assert_eq!(evidence.selected_eigenvalue.requested_index, 1);
        assert!(evidence
            .deficiency
            .concentration_eigenvalue
            .contains(&q(7, 13)));
        assert!(evidence.deficiency.deficiency.lower() > &q(0, 1));
        assert!(evidence.deficiency.deficiency.upper() < &q(1, 1));
        assert!(!evidence.deficiency.exact_finite);
    }

    #[test]
    fn residual_to_gap_is_fail_closed() {
        let reliable = residual_to_certified_gap(q(1, 1000), q(1, 10)).unwrap();
        assert!(matches!(
            reliable,
            ResidualGapConclusion::Reliable {
                sin_angle_upper,
                ..
            } if sin_angle_upper == q(1, 100)
        ));
        assert!(matches!(
            residual_to_certified_gap(q(1, 5), q(1, 10)).unwrap(),
            ResidualGapConclusion::Inconclusive { .. }
        ));
    }

    #[test]
    fn prolate_weil_state_comparison_covers_all_three_spaces() {
        let precision = 192;
        let vector = |values: &[i32]| {
            values
                .iter()
                .map(|value| Float::with_val(precision, *value))
                .collect::<Vec<_>>()
        };
        let bounds = vec![
            ActiveTruncationBound {
                kind: ComparisonTruncationKind::ValueSpace,
                upper_bound: Float::with_val(precision, Float::parse("1e-20").unwrap()),
                source: "sample-grid tail theorem".to_owned(),
            },
            ActiveTruncationBound {
                kind: ComparisonTruncationKind::CoefficientSpace,
                upper_bound: Float::with_val(precision, Float::parse("2e-20").unwrap()),
                source: "basis-tail estimate".to_owned(),
            },
            ActiveTruncationBound {
                kind: ComparisonTruncationKind::FormNorm,
                upper_bound: Float::with_val(precision, Float::parse("3e-20").unwrap()),
                source: "operator-tail estimate".to_owned(),
            },
        ];
        let report = compare_prolate_weil_states_hp(
            &vector(&[1, 0]),
            &vector(&[-3, -4]),
            &vector(&[1, 0]),
            &vector(&[-3, -4]),
            &vector(&[2, 0, 0, 5]),
            bounds,
            precision,
        )
        .unwrap();

        let overlap = Float::with_val(precision, Float::parse("0.6").unwrap());
        let tolerance = Float::with_val(precision, Float::parse("1e-50").unwrap());
        assert!((report.value_space_overlap - &overlap).abs() < tolerance);
        assert!((report.coefficient_overlap - overlap).abs() < tolerance);
        assert!(report.value_space_residual > 0);
        assert!(report.coefficient_residual > 0);
        assert!(report.prolate_eigen_residual > 0);
        assert_eq!(report.weil_eigen_residual, 0);
        assert!(report.form_norm_difference > 0);
        assert_eq!(report.truncation_bounds.len(), 3);
    }
}
