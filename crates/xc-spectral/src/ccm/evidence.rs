// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Finite CCM evidence: exact algebra, conditional bounds and point diagnostics.
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
#[cfg(test)]
use rug::float::Constant;
use rug::{Float, Rational};
use xc_numerics::interval::RationalInterval;

mod state_comparison;
pub use state_comparison::{
    compare_prolate_weil_states_hp, ActiveTruncationBound, ComparisonTruncationKind,
    ProlateWeilStateComparisonHp,
};
mod asymptotic;
pub use asymptotic::{
    prolate_chi2_deficiency_asymptotic, prolate_chi2_log_deficiency_asymptotic,
    prolate_log_deficiency_asymptotic, try_prolate_chi2_deficiency_asymptotic,
    try_prolate_deficiency_asymptotic, EvenProlateMode,
};
mod tail_budget;
pub use tail_budget::{
    finite_cutoff_decision, finite_cutoff_interval_decision, try_finite_cutoff_decision,
    ArchimedeanTailBudget, FiniteCutoffDecision,
};

/// Exact interval propagation of `1-sqrt(nu)` for supplied nu in `[0,1]`.
/// Identifying nu with a particular prolate mode, and establishing its source
/// enclosure, are external premises. This record certifies the algebra only.
///
/// ```compile_fail
/// use rug::Rational;
/// use xc_numerics::interval::RationalInterval;
/// use xc_spectral::ccm::evidence::CertifiedProlateDeficiency;
/// let mut value = CertifiedProlateDeficiency::from_concentration_enclosure(
///     RationalInterval::point(Rational::from((1, 4))), 64).unwrap();
/// value.deficiency = RationalInterval::point(Rational::from(0));
/// ```
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CertifiedProlateDeficiency {
    /// Certified enclosure of a concentration value `nu = |chi_n|^2`.
    concentration_eigenvalue: RationalInterval,
    /// Certified enclosure of the positive singular value `|chi_n|`.
    angular_eigenvalue: RationalInterval,
    /// Certified enclosure of the singular-value deficit `1 - |chi_n|`.
    deficiency: RationalInterval,
    /// Whether the finite deficiency was represented exactly by the dyadic
    /// square-root grid rather than by a nonzero-width enclosure.
    exact_finite: bool,
    fraction_bits: u32,
}

impl CertifiedProlateDeficiency {
    /// Propagate a certified enclosure of `nu` through the positive square
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
    pub fn concentration_eigenvalue(&self) -> &RationalInterval {
        &self.concentration_eigenvalue
    }
    /// Positive singular value; the historical name does not encode a
    /// signed Fourier phase. Use `signed_fourier_eigenvalue` for that phase.
    pub fn angular_eigenvalue(&self) -> &RationalInterval {
        &self.angular_eigenvalue
    }
    #[doc(hidden)]
    pub fn singular_value(&self) -> &RationalInterval {
        &self.angular_eigenvalue
    }
    /// Apply the caller-declared full-mode Fourier phase. Identifying the
    /// supplied concentration enclosure with this mode is an external premise.
    #[doc(hidden)]
    pub fn signed_fourier_eigenvalue(&self, mode: EvenProlateMode) -> RationalInterval {
        if mode.fourier_sign() > 0 {
            self.angular_eigenvalue.clone()
        } else {
            self.angular_eigenvalue.neg()
        }
    }
    pub fn deficiency(&self) -> &RationalInterval {
        &self.deficiency
    }
    pub fn exact_finite(&self) -> bool {
        self.exact_finite
    }
    pub fn fraction_bits(&self) -> u32 {
        self.fraction_bits
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
    /// Zero-based ASCENDING index of the supplied finite matrix.
    /// Conventional prolate concentration modes are descending: chi_2^2 is
    /// descending even-sector index 2 (full even mode 4), at C=2*pi*lambda^2.
    /// Identifying a matrix index with that mode remains a separate premise.
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
/// Establishing that the supplied matrix and index represent the intended
/// continuum prolate mode (including discretization error) is a separate premise.
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
            try_prolate_chi2_deficiency_asymptotic(lambda_squared, precision_bits)?;
        let finite_difference = measured_weil_plunge.sub(&finite_prolate.deficiency);
        Ok(Self {
            finite_prolate,
            asymptotic_predictor,
            measured_weil_plunge,
            finite_difference,
        })
    }
}

/// Explicit indices for a proposed comparison. No correspondence between a
/// Weil branch and a prolate mode is inferred or certified by this record.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[doc(hidden)]
pub struct ProlateWeilComparisonIndices {
    pub prolate_mode: EvenProlateMode,
    pub weil_parity: super::hp::CcmParity,
    /// Zero-based algebraic index WITHIN `weil_parity`, not the full matrix.
    pub weil_spectral_index: usize,
}

/// Mode-labelled extension of the legacy n=4 comparison. The caller must bind
/// the concentration interval and measured Weil interval to these declared
/// source indices. Neither interval's source/continuum validity follows from
/// this algebraic comparison. The measured interval keeps its original sign.
#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct IndexedProlateWeilComparison {
    pub indices: ProlateWeilComparisonIndices,
    pub lambda_squared: u64,
    pub comparison: ProlateWeilComparison,
}

impl IndexedProlateWeilComparison {
    pub fn new(
        indices: ProlateWeilComparisonIndices,
        lambda_squared: u64,
        finite_prolate: CertifiedProlateDeficiency,
        measured_weil_plunge: RationalInterval,
        precision_bits: u32,
    ) -> Result<Self> {
        if lambda_squared <= 1 {
            bail!("prolate comparison requires lambda^2 > 1");
        }
        let asymptotic_predictor = try_prolate_deficiency_asymptotic(
            indices.prolate_mode,
            lambda_squared,
            precision_bits,
        )?;
        let finite_difference = measured_weil_plunge.sub(finite_prolate.deficiency());
        Ok(Self {
            indices,
            lambda_squared,
            comparison: ProlateWeilComparison {
                finite_prolate,
                asymptotic_predictor,
                measured_weil_plunge,
                finite_difference,
            },
        })
    }
}

/// Conditional residual/separation bound for a unit vector v, a self-adjoint
/// operator A, a scalar mu and an identified one-dimensional target eigenspace.
/// If ||Av-mu*v|| <= r and every unwanted eigenvalue has distance at least delta
/// from mu, then sin(angle) <= r/delta. The spectral theorem proves this by
/// bounding every unwanted component in the residual norm.
///
/// A gap between the target eigenvalue and its neighbors is NOT this delta.
/// It must first be reduced by a certified |mu-lambda_target| error. Neither
/// helper here establishes normalization, self-adjointness, target identity,
/// residual validity or spectral exclusion; those remain explicit premises.
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

/// Conditional ratio using a separately certified distance from mu to the
/// unwanted spectrum. `gap_lower` in the result means this separation, never
/// a raw spacing between eigenvalues.
pub fn residual_to_complement_separation(
    residual_upper: Rational,
    complement_separation_lower: Rational,
) -> Result<ResidualGapConclusion> {
    let gap_lower = complement_separation_lower;
    if residual_upper < 0 {
        bail!("residual upper bound must be non-negative");
    }
    if gap_lower <= 0 {
        bail!("certified distance from mu to the unwanted spectrum must be positive");
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

/// Retired ambiguous compatibility entry point. A raw eigenvalue spacing does
/// not determine the residual separation needed by the angle theorem. Always
/// returns an error; use the explicitly named separation or gap/error API.
pub fn residual_to_certified_gap(
    _residual_upper: Rational,
    _gap_lower: Rational,
) -> Result<ResidualGapConclusion> {
    bail!("ambiguous gap premise: use residual_to_complement_separation or residual_to_eigenvalue_gap with a certified eigenvalue error")
}

/// Convert a target-to-complement eigenvalue gap g using a separately certified
/// error e>=|mu-lambda_target|. Triangle inequality gives delta>=g-e. Failure
/// to establish a positive delta returns an error; r>=delta is inconclusive.
pub fn residual_to_eigenvalue_gap(
    residual_upper: Rational,
    eigenvalue_gap_lower: Rational,
    eigenvalue_error_upper: Rational,
) -> Result<ResidualGapConclusion> {
    if eigenvalue_error_upper < 0 || eigenvalue_gap_lower <= eigenvalue_error_upper {
        bail!("positive complement separation requires gap > nonnegative eigenvalue error");
    }
    residual_to_complement_separation(
        residual_upper,
        eigenvalue_gap_lower - eigenvalue_error_upper,
    )
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
        assert!(budget.upper_bound() > &0);
        assert_eq!(
            finite_cutoff_decision(&Float::with_val(precision, 1), &budget),
            FiniteCutoffDecision::CutoffFreePositive
        );
        let mut deep_negative = budget.upper_bound().clone();
        deep_negative *= -2i32;
        assert_eq!(
            finite_cutoff_decision(&deep_negative, &budget),
            FiniteCutoffDecision::CutoffFreeNegative
        );
        let mut shallow_negative = budget.upper_bound().clone();
        shallow_negative /= -2i32;
        assert_eq!(
            finite_cutoff_decision(&shallow_negative, &budget),
            FiniteCutoffDecision::InconclusiveTailBand
        );
    }

    /// The prefactor is half of Fuchs' constant -- Fuchs governs `1 - lambda_4`
    /// and `chi_2^2 = lambda_4` halves it -- and its base-ten logarithm is the
    /// published `C_0`. Pinning it here keeps the predictor tied to the
    /// literature rather than to whatever the code happened to contain.
    #[test]
    fn prolate_asymptotic_prefactor_matches_the_published_c0() {
        let prec = 192;
        // Strip the exponential: at lambda^2 = 1 the exponent is -4*pi.
        let predictor = prolate_chi2_deficiency_asymptotic(1, prec);
        let restored =
            predictor * Float::with_val(prec, Float::with_val(prec, Constant::Pi) * 4u32).exp();
        let log10 = restored.ln() / Float::with_val(prec, 10u32).ln();
        let expected = Float::with_val(prec, Float::parse("6.373563046").unwrap());
        let error = Float::with_val(prec, &log10 - &expected).abs();
        assert!(
            error < Float::with_val(prec, Float::parse("1e-9").unwrap()),
            "log10 prefactor {log10:?} does not match the published C_0"
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
        let reliable = residual_to_complement_separation(q(1, 1000), q(1, 10)).unwrap();
        assert!(matches!(
            reliable,
            ResidualGapConclusion::Reliable {
                sin_angle_upper,
                ..
            } if sin_angle_upper == q(1, 100)
        ));
        assert!(matches!(
            residual_to_complement_separation(q(1, 5), q(1, 10)).unwrap(),
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
