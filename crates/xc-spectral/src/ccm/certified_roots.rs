//! Certified roots and independent counts for finite CCM secular functions.
//!
//! A stored finite state defines
//! `R(z) = sum_j weight_j / (z - pole_j)`.  Interval Newton supplies root
//! enclosures and uniqueness.  When every nonzero residue has one strict
//! sign, monotonicity and the one-sided pole limits give an independent count
//! of exactly one root in every open interval between adjacent poles.

use anyhow::{bail, Context, Result};
#[cfg(feature = "arb")]
use rug::ops::Pow;
use rug::{Float, Integer, Rational};
use serde::{Deserialize, Serialize};
use xc_cache::ContentDigest;
use xc_core::ConvergenceTableRow;
use xc_numerics::interval::{
    certify_polynomial_zero_count_on_rectangle, differentiate_complex_polynomial,
    evaluate_complex_polynomial_ball, evaluate_complex_polynomial_exact,
    evaluate_real_polynomial_exact, exact_sturm_isolate_roots, exact_sturm_root_count,
    ComplexRational, ComplexRationalBall, PolynomialContourCount, RationalContourRectangle,
};
use xc_numerics::mpfr_interval::MpfrInterval;
use xc_root::{
    interval_newton_hp, IntervalNewtonOptions, IntervalRootCertificate, IntervalRootStatus,
    RealIntervalFunctionHp, RootError,
};

#[derive(Clone, Debug)]
pub struct CertifiedSecularFunction {
    poles: Vec<MpfrInterval>,
    weights: Vec<MpfrInterval>,
}

impl CertifiedSecularFunction {
    pub fn from_integer_ccm_state(
        integer_cutoff_c: u64,
        modes: usize,
        weights: &[Float],
        precision_bits: u32,
    ) -> Result<Self> {
        if integer_cutoff_c <= 1 || weights.len() != 2 * modes + 1 {
            bail!("CCM secular source requires c > 1 and exactly 2N+1 weights");
        }
        let log_c = Float::with_val(precision_bits, integer_cutoff_c).ln();
        let mut spacing = Float::with_val(precision_bits, rug::float::Constant::Pi);
        spacing *= 2;
        spacing /= log_c;
        let poles = (-(modes as i64)..=(modes as i64))
            .map(|index| {
                let mut pole = spacing.clone();
                pole *= index;
                pole
            })
            .collect::<Vec<_>>();
        Self::from_point_data(&poles, weights, precision_bits)
    }

    pub fn from_point_data(
        poles: &[Float],
        weights: &[Float],
        precision_bits: u32,
    ) -> Result<Self> {
        if poles.is_empty() || poles.len() != weights.len() {
            bail!("finite secular source needs equal nonempty pole and weight arrays");
        }
        let poles: Vec<_> = poles
            .iter()
            .map(|value| MpfrInterval::point(Float::with_val(precision_bits, value)))
            .collect();
        let weights: Vec<_> = weights
            .iter()
            .map(|value| MpfrInterval::point(Float::with_val(precision_bits, value)))
            .collect();
        Self::from_intervals(poles, weights)
    }

    pub fn from_intervals(poles: Vec<MpfrInterval>, weights: Vec<MpfrInterval>) -> Result<Self> {
        if poles.is_empty() || poles.len() != weights.len() {
            bail!("finite secular source needs equal nonempty pole and weight arrays");
        }
        let precision = poles[0].precision();
        if poles
            .iter()
            .chain(&weights)
            .any(|value| value.precision() != precision)
        {
            bail!("finite secular pole and weight intervals must share one precision");
        }
        for pair in poles.windows(2) {
            if pair[0].upper() >= pair[1].lower() {
                bail!("finite secular pole intervals must be strictly ordered and disjoint");
            }
        }
        if weights.iter().any(MpfrInterval::contains_zero) {
            bail!("finite secular residue intervals must exclude zero");
        }
        Ok(Self { poles, weights })
    }

    pub fn poles(&self) -> &[MpfrInterval] {
        &self.poles
    }

    pub fn precision_bits(&self) -> u32 {
        self.poles[0].precision()
    }

    pub fn isolate(
        &self,
        candidate: &MpfrInterval,
        options: &IntervalNewtonOptions,
    ) -> Result<IntervalRootCertificate> {
        if self.overlaps_pole(candidate) {
            bail!("candidate root interval overlaps a finite secular pole");
        }
        interval_newton_hp(self, candidate, options).map_err(anyhow::Error::from)
    }

    /// Discover and certify the unique root in one adjacent-pole interval.
    ///
    /// The bracket is obtained only from pole geometry and interval signs;
    /// no reference ordinate or point-solver seed participates.
    pub fn certify_pole_interval(
        &self,
        left_pole: usize,
        bisection_steps: usize,
        options: &IntervalNewtonOptions,
    ) -> Result<IntervalRootCertificate> {
        if left_pole + 1 >= self.poles.len() || bisection_steps == 0 {
            bail!("pole-interval certification needs an adjacent pole pair and bisection steps");
        }
        let precision = self.precision_bits();
        let left_boundary = self.poles[left_pole].upper();
        let right_boundary = self.poles[left_pole + 1].lower();
        let mut margin = Float::with_val(precision, right_boundary - left_boundary);
        margin /= 1024;
        let mut lower = Float::with_val(precision, left_boundary + &margin);
        let mut upper = Float::with_val(precision, right_boundary - &margin);
        let mut lower_value = self
            .evaluate_interval(&MpfrInterval::point(lower.clone()))
            .map_err(anyhow::Error::from)?;
        let upper_value = self
            .evaluate_interval(&MpfrInterval::point(upper.clone()))
            .map_err(anyhow::Error::from)?;
        let opposite = (lower_value.is_strictly_positive() && upper_value.upper() < &0)
            || (lower_value.upper() < &0 && upper_value.is_strictly_positive());
        if !opposite {
            bail!("finite secular pole interval lacks strict opposite endpoint signs");
        }
        for _ in 0..bisection_steps {
            let mut midpoint = lower.clone();
            midpoint += &upper;
            midpoint /= 2;
            let midpoint_value = self
                .evaluate_interval(&MpfrInterval::point(midpoint.clone()))
                .map_err(anyhow::Error::from)?;
            if midpoint_value.contains_zero() {
                break;
            }
            let same_as_lower =
                midpoint_value.is_strictly_positive() == lower_value.is_strictly_positive();
            if same_as_lower {
                lower = midpoint;
                lower_value = midpoint_value;
            } else {
                upper = midpoint;
            }
        }
        self.isolate(&MpfrInterval::new(lower, upper)?, options)
    }

    fn overlaps_pole(&self, argument: &MpfrInterval) -> bool {
        self.poles
            .iter()
            .any(|pole| argument.lower() <= pole.upper() && argument.upper() >= pole.lower())
    }

    pub fn monotone_count(&self) -> Result<SecularCountCertificate> {
        self.monotone_count_between_poles(0, self.poles.len() - 1)
    }

    pub fn monotone_count_between_poles(
        &self,
        first_pole: usize,
        last_pole: usize,
    ) -> Result<SecularCountCertificate> {
        if first_pole >= last_pole || last_pole >= self.poles.len() {
            bail!("monotone count needs an ordered in-range pole span");
        }
        let positive = self.weights.iter().all(MpfrInterval::is_strictly_positive);
        let negative = self.weights.iter().all(|weight| weight.upper() < &0);
        if !positive && !negative {
            bail!("independent monotone count requires all residue intervals to have one sign");
        }
        Ok(SecularCountCertificate {
            pole_count: last_pole - first_pole + 1,
            open_intervals_counted: last_pole - first_pole,
            certified_root_count: last_pole - first_pole,
            residue_sign: if positive { "positive" } else { "negative" }.to_owned(),
            method: "same-sign-residue monotonicity and one-sided pole limits".to_owned(),
            square_free: true,
        })
    }

    fn exact_numerator_data(&self) -> Result<(Vec<Rational>, Vec<Rational>)> {
        let point = |interval: &MpfrInterval| -> Result<Rational> {
            if interval.lower() != interval.upper() {
                bail!("exact numerator count requires point poles and residues");
            }
            interval
                .lower()
                .to_rational()
                .context("convert finite MPFR point to an exact rational")
        };
        let poles = self.poles.iter().map(point).collect::<Result<Vec<_>>>()?;
        let weights = self.weights.iter().map(point).collect::<Result<Vec<_>>>()?;
        // Build P(x) = product_j (x - p_j) once, then obtain each excluded
        // product P(x)/(x-p_j) by exact synthetic division. The former
        // implementation rebuilt n-1 factors for every residue (O(n^3)
        // rational operations); this route is O(n^2) and produces the same
        // exact ascending coefficient vector.
        let mut pole_product = vec![Rational::from((1, 1))];
        for pole in &poles {
            let mut next = vec![Rational::from((0, 1)); pole_product.len() + 1];
            for (power, coefficient) in pole_product.iter().enumerate() {
                let mut constant = coefficient.clone();
                constant *= pole;
                next[power] -= constant;
                next[power + 1] += coefficient;
            }
            pole_product = next;
        }
        let mut numerator = vec![Rational::from((0, 1)); poles.len()];
        for (excluded, weight) in weights.iter().enumerate() {
            let pole = &poles[excluded];
            let mut term = vec![Rational::from((0, 1)); poles.len()];
            term[poles.len() - 1] = pole_product[poles.len()].clone();
            for power in (1..poles.len()).rev() {
                let mut carried = term[power].clone();
                carried *= pole;
                term[power - 1] = pole_product[power].clone();
                term[power - 1] += carried;
            }
            let mut remainder = term[0].clone();
            remainder *= pole;
            remainder += &pole_product[0];
            if remainder != 0 {
                bail!("exact secular pole-product synthetic division left a remainder");
            }
            for (power, coefficient) in term.iter().enumerate() {
                let mut contribution = coefficient.clone();
                contribution *= weight;
                numerator[power] += contribution;
            }
        }
        Ok((poles, numerator))
    }

    /// Construct the monic finite entire function obtained by cancelling the
    /// known secular poles. No reference roots or sampled contour values are
    /// used: every coefficient is derived exactly from the stored point poles
    /// and residues.
    pub fn normalized_finite_entire_function(&self) -> Result<FiniteEntireFunction> {
        let (_, numerator) = self.exact_numerator_data()?;
        FiniteEntireFunction::from_real_coefficients_monic(numerator)
    }

    pub fn exact_numerator_count_between_poles(
        &self,
        first_pole: usize,
        last_pole: usize,
    ) -> Result<SecularCountCertificate> {
        if first_pole >= last_pole || last_pole >= self.poles.len() {
            bail!("exact numerator count needs an ordered in-range pole span");
        }
        let (poles, numerator) = self.exact_numerator_data()?;
        let count = exact_sturm_root_count(
            &numerator,
            poles[first_pole].clone(),
            poles[last_pole].clone(),
        )?;
        Ok(SecularCountCertificate {
            pole_count: last_pole - first_pole + 1,
            open_intervals_counted: last_pole - first_pole,
            certified_root_count: count.distinct_real_roots,
            residue_sign: "not_required".to_owned(),
            method: "exact-rational secular numerator Sturm sequence".to_owned(),
            square_free: count.square_free,
        })
    }

    /// Reference-free discovery by exact Sturm subdivision followed by an
    /// independent interval-Newton existence/uniqueness proof for each root.
    pub fn certify_exact_numerator_window(
        &self,
        first_pole: usize,
        last_pole: usize,
        isolation_bits: u32,
        options: &IntervalNewtonOptions,
    ) -> Result<SecularWindowCertificate> {
        if isolation_bits < 16 {
            bail!("exact numerator isolation requires at least 16 width bits");
        }
        let (poles, numerator) = self.exact_numerator_data()?;
        if first_pole >= last_pole || last_pole >= poles.len() {
            bail!("exact numerator window needs an ordered in-range pole span");
        }
        let denominator = Integer::from(1) << isolation_bits;
        let target_width = Rational::from((Integer::from(1), denominator));
        let candidates = exact_sturm_isolate_roots(
            &numerator,
            poles[first_pole].clone(),
            poles[last_pole].clone(),
            target_width,
            (isolation_bits as usize).saturating_mul(4),
        )?;
        let roots = candidates
            .iter()
            .map(|candidate| {
                let lower = MpfrInterval::from_rational(candidate.lower(), self.precision_bits());
                let upper = MpfrInterval::from_rational(candidate.upper(), self.precision_bits());
                let interval = MpfrInterval::new(lower.lower().clone(), upper.upper().clone())?;
                self.isolate(&interval, options)
            })
            .collect::<Result<Vec<_>>>()?;
        let count = self.exact_numerator_count_between_poles(first_pole, last_pole)?;
        let reconciliation = reconcile_complete_window(&roots, &count)?;
        Ok(SecularWindowCertificate {
            roots,
            count,
            reconciliation,
        })
    }

    /// Production-scale exact numerator count using the system FLINT
    /// engine. Rational MPFR source points are transferred losslessly; FLINT
    /// clears denominators, isolates every root in a certified Arb complex
    /// ball, and classifies only balls strictly separated from the rational
    /// window boundaries.
    #[cfg(feature = "arb")]
    pub fn exact_flint_numerator_count_between_poles(
        &self,
        first_pole: usize,
        last_pole: usize,
    ) -> Result<SecularCountCertificate> {
        if first_pole >= last_pole || last_pole >= self.poles.len() {
            bail!("FLINT numerator count needs an ordered in-range pole span");
        }
        let (poles, _) = self.exact_numerator_data()?;
        let mut count =
            self.exact_flint_numerator_count_in_window(&poles[first_pole], &poles[last_pole])?;
        count.pole_count = last_pole - first_pole + 1;
        count.open_intervals_counted = last_pole - first_pole;
        Ok(count)
    }

    /// Count all distinct real roots in an arbitrary exact rational window.
    /// This is the primitive used for height-window discovery and ordinal
    /// assignment; neither endpoint may be derived from a reference zero in
    /// an independent run.
    #[cfg(feature = "arb")]
    pub fn exact_flint_numerator_count_in_window(
        &self,
        lower: &Rational,
        upper: &Rational,
    ) -> Result<SecularCountCertificate> {
        if lower >= upper {
            bail!("FLINT numerator count requires lower < upper");
        }
        let (_, numerator) = self.exact_numerator_data()?;
        let (count, square_free) =
            crate::ccm::arb_bridge::rational_polynomial_root_count(&numerator, lower, upper)?;
        Ok(SecularCountCertificate {
            pole_count: self
                .poles
                .iter()
                .filter(|pole| {
                    pole.lower()
                        .to_rational()
                        .is_some_and(|value| &value >= lower && &value <= upper)
                })
                .count(),
            open_intervals_counted: 0,
            certified_root_count: count,
            residue_sign: "not_required".to_owned(),
            method: "FLINT/Arb certified integer-polynomial complex-root isolation and real-window classification"
                .to_owned(),
            square_free,
        })
    }

    /// Reference-free production isolation through exact FLINT/Arb root balls
    /// followed by the independent interval-Newton proof for the original
    /// rational secular function.
    #[cfg(feature = "arb")]
    pub fn certify_flint_numerator_window(
        &self,
        first_pole: usize,
        last_pole: usize,
        isolation_bits: u32,
        options: &IntervalNewtonOptions,
    ) -> Result<SecularWindowCertificate> {
        if isolation_bits < 16 || isolation_bits >= self.precision_bits() {
            bail!("FLINT numerator isolation width must fit below the source precision");
        }
        let (poles, _) = self.exact_numerator_data()?;
        if first_pole >= last_pole || last_pole >= poles.len() {
            bail!("FLINT numerator window needs an ordered in-range pole span");
        }
        self.certify_flint_numerator_rational_window(
            &poles[first_pole],
            &poles[last_pole],
            isolation_bits,
            options,
        )
    }

    /// Isolate and certify every real root in an arbitrary exact rational
    /// window. Completeness comes from the exact numerator count, while each
    /// retained root receives an interval-Newton existence/uniqueness proof.
    #[cfg(feature = "arb")]
    pub fn certify_flint_numerator_rational_window(
        &self,
        lower_bound: &Rational,
        upper_bound: &Rational,
        isolation_bits: u32,
        options: &IntervalNewtonOptions,
    ) -> Result<SecularWindowCertificate> {
        let (_, numerator) = self.exact_numerator_data()?;
        if lower_bound >= upper_bound {
            bail!("FLINT numerator window requires lower < upper");
        }
        let (mut candidates, square_free) = crate::ccm::arb_bridge::rational_polynomial_real_roots(
            &numerator,
            lower_bound,
            upper_bound,
            self.precision_bits(),
        )?;
        if !square_free {
            bail!("FLINT numerator isolation requires a square-free polynomial");
        }
        let target_width = Float::with_val(self.precision_bits(), 2).pow(-(isolation_bits as i32));
        for candidate in &candidates {
            let width =
                Float::with_val(self.precision_bits(), candidate.upper() - candidate.lower());
            if width > target_width {
                bail!("Arb root enclosure did not meet the requested absolute width");
            }
        }
        candidates.sort_by(|left, right| {
            left.lower()
                .partial_cmp(right.lower())
                .expect("certified Arb root endpoints are finite")
        });
        let count = self.exact_flint_numerator_count_in_window(lower_bound, upper_bound)?;
        if count.certified_root_count != candidates.len() || count.square_free != square_free {
            bail!("FLINT root isolation disagrees with its independent exact count");
        }
        let roots = candidates
            .into_iter()
            .map(|candidate| {
                let mut lower = candidate.lower().clone();
                lower -= &target_width;
                let mut upper = candidate.upper().clone();
                upper += &target_width;
                self.isolate(&MpfrInterval::new(lower, upper)?, options)
            })
            .collect::<Result<Vec<_>>>()?;
        let reconciliation = reconcile_complete_window(&roots, &count)?;
        if !reconciliation.complete {
            bail!(
                "FLINT/Arb production window failed interval-Newton reconciliation: {}",
                reconciliation.reason
            );
        }
        Ok(SecularWindowCertificate {
            roots,
            count,
            reconciliation,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FiniteEntireFunction {
    coefficients_ascending: Vec<ComplexRational>,
    derivative_coefficients_ascending: Vec<ComplexRational>,
}

impl FiniteEntireFunction {
    pub fn from_real_coefficients_monic(mut coefficients: Vec<Rational>) -> Result<Self> {
        while coefficients.len() > 1 && coefficients.last().is_some_and(|value| value == &0) {
            coefficients.pop();
        }
        let Some(leading) = coefficients.last().cloned() else {
            bail!("finite entire function requires at least one coefficient");
        };
        if leading == 0 {
            bail!("finite entire function cannot be identically zero");
        }
        for coefficient in &mut coefficients {
            *coefficient /= &leading;
        }
        let coefficients_ascending = coefficients
            .into_iter()
            .map(|real| ComplexRational {
                real,
                imaginary: Rational::from((0, 1)),
            })
            .collect::<Vec<_>>();
        let derivative_coefficients_ascending =
            differentiate_complex_polynomial(&coefficients_ascending)?;
        Ok(Self {
            coefficients_ascending,
            derivative_coefficients_ascending,
        })
    }

    pub fn coefficients(&self) -> &[ComplexRational] {
        &self.coefficients_ascending
    }

    pub fn derivative_coefficients(&self) -> &[ComplexRational] {
        &self.derivative_coefficients_ascending
    }

    pub fn degree(&self) -> usize {
        self.coefficients_ascending.len().saturating_sub(1)
    }

    pub fn evaluate_real_exact(&self, argument: &Rational) -> Result<Rational> {
        let coefficients = self
            .coefficients_ascending
            .iter()
            .map(|coefficient| coefficient.real.clone())
            .collect::<Vec<_>>();
        Ok(evaluate_real_polynomial_exact(&coefficients, argument)?)
    }

    pub fn evaluate_exact(&self, argument: &ComplexRational) -> Result<ComplexRational> {
        Ok(evaluate_complex_polynomial_exact(
            &self.coefficients_ascending,
            argument,
        )?)
    }

    pub fn evaluate_ball(&self, argument: &ComplexRationalBall) -> Result<ComplexRationalBall> {
        Ok(evaluate_complex_polynomial_ball(
            &self.coefficients_ascending,
            argument,
        )?)
    }

    pub fn evaluate_derivative_exact(&self, argument: &ComplexRational) -> Result<ComplexRational> {
        Ok(evaluate_complex_polynomial_exact(
            &self.derivative_coefficients_ascending,
            argument,
        )?)
    }

    pub fn evaluate_derivative_ball(
        &self,
        argument: &ComplexRationalBall,
    ) -> Result<ComplexRationalBall> {
        Ok(evaluate_complex_polynomial_ball(
            &self.derivative_coefficients_ascending,
            argument,
        )?)
    }

    pub fn certify_zero_count(
        &self,
        rectangle: RationalContourRectangle,
        maximum_subdivision_depth: usize,
    ) -> Result<PolynomialContourCount> {
        Ok(certify_polynomial_zero_count_on_rectangle(
            &self.coefficients_ascending,
            rectangle,
            maximum_subdivision_depth,
        )?)
    }
}

impl RealIntervalFunctionHp for CertifiedSecularFunction {
    fn evaluate_interval(&self, argument: &MpfrInterval) -> Result<MpfrInterval, RootError> {
        if argument.precision() != self.precision_bits() {
            return Err(RootError::InvalidConfiguration(
                "secular argument precision differs from source precision".to_owned(),
            ));
        }
        if self.overlaps_pole(argument) {
            return Err(RootError::PoleCollision(
                "secular interval evaluation overlaps a pole".to_owned(),
            ));
        }
        let mut value = MpfrInterval::from_i64(0, self.precision_bits());
        for (pole, weight) in self.poles.iter().zip(&self.weights) {
            value = value.add(
                &weight
                    .div(&argument.sub(pole))
                    .map_err(|error| RootError::Evaluation(error.to_string()))?,
            );
        }
        Ok(value)
    }

    fn derivative_interval(&self, argument: &MpfrInterval) -> Result<MpfrInterval, RootError> {
        if argument.precision() != self.precision_bits() {
            return Err(RootError::InvalidConfiguration(
                "secular argument precision differs from source precision".to_owned(),
            ));
        }
        if self.overlaps_pole(argument) {
            return Err(RootError::PoleCollision(
                "secular derivative interval overlaps a pole".to_owned(),
            ));
        }
        let mut derivative = MpfrInterval::from_i64(0, self.precision_bits());
        for (pole, weight) in self.poles.iter().zip(&self.weights) {
            derivative = derivative.sub(
                &weight
                    .div(&argument.sub(pole).square())
                    .map_err(|error| RootError::Evaluation(error.to_string()))?,
            );
        }
        Ok(derivative)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SecularCountCertificate {
    pub pole_count: usize,
    pub open_intervals_counted: usize,
    pub certified_root_count: usize,
    pub residue_sign: String,
    pub method: String,
    pub square_free: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RootCountReconciliation {
    pub complete: bool,
    pub isolated_root_count: usize,
    pub independently_counted_roots: usize,
    pub ordered_and_disjoint: bool,
    pub reason: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SecularWindowCertificate {
    pub roots: Vec<IntervalRootCertificate>,
    pub count: SecularCountCertificate,
    pub reconciliation: RootCountReconciliation,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FirstPositiveCcmRootOptions {
    pub integer_cutoff_c: u64,
    pub modes: usize,
    pub requested_roots: usize,
    pub precision_bits: u32,
    pub pole_interval_bisection_steps: usize,
    pub interval_newton: IntervalNewtonOptions,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FirstPositiveCcmRootCertificate {
    pub schema_version: u32,
    pub integer_cutoff_c: u64,
    pub modes: usize,
    pub requested_roots: usize,
    pub precision_bits: u32,
    pub source_weights: Vec<String>,
    pub source_weights_digest: ContentDigest,
    pub roots: Vec<IntervalRootCertificate>,
    pub independent_count: SecularCountCertificate,
    pub reconciliation: RootCountReconciliation,
    pub reference_seeds_used: bool,
    pub discovery_method: String,
    pub count_method: String,
    pub finite_model_statement: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProductionFirstPositiveCcmRootCertificate {
    pub schema_version: u32,
    pub integer_cutoff_c: u64,
    pub modes: usize,
    pub requested_roots: usize,
    pub precision_bits: u32,
    pub isolation_bits: u32,
    pub interval_newton: IntervalNewtonOptions,
    pub first_pole: usize,
    pub last_pole: usize,
    pub source_weights: Vec<String>,
    pub source_weights_digest: ContentDigest,
    pub window: SecularWindowCertificate,
    pub preceding_window_count: SecularCountCertificate,
    pub following_window_count: SecularCountCertificate,
    pub reference_seeds_used: bool,
    pub discovery_method: String,
    pub count_method: String,
    pub finite_model_statement: String,
}

/// Reference-free root selection for a finite production CCM source.
/// Positive indices are assigned by exact cumulative counts from the zero
/// pole; they are never copied from an external zeta-zero table.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "target")]
pub enum IndependentCcmRootTarget {
    Prefix { count: usize },
    IndexRange { first: usize, last: usize },
    PositiveHeightWindow { lower: String, upper: String },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FiniteSourceCertificationScope {
    /// Exact certification of the stored dyadic/MPFR point source. This does
    /// not claim that Tau/eigenvector uncertainty was propagated.
    ExactStoredPointSource,
    /// Reserved for a certificate that propagates interval Tau, spectral-gap,
    /// eigenvector, normalization, and residue uncertainty end to end.
    IntervalFiniteCcmOperator,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProductionIndependentCcmRootCertificate {
    pub schema_version: u32,
    pub integer_cutoff_c: u64,
    pub modes: usize,
    pub precision_bits: u32,
    pub isolation_bits: u32,
    pub interval_newton: IntervalNewtonOptions,
    pub target: IndependentCcmRootTarget,
    /// Exact rational discovery boundaries used by FLINT.
    pub lower_bound: String,
    pub upper_bound: String,
    /// Number of positive finite-source roots strictly before `lower_bound`.
    pub positive_roots_before_window: usize,
    /// Offset and length of the requested roots within `window.roots`.
    pub selected_root_offset: usize,
    pub selected_root_count: usize,
    pub first_selected_positive_index: Option<usize>,
    pub last_selected_positive_index: Option<usize>,
    pub source_weights: Vec<String>,
    pub source_weights_digest: ContentDigest,
    pub source_certification_scope: FiniteSourceCertificationScope,
    pub window: SecularWindowCertificate,
    pub reference_seeds_used: bool,
    pub discovery_method: String,
    pub count_method: String,
    pub finite_model_statement: String,
}

impl ProductionIndependentCcmRootCertificate {
    pub fn selected_roots(&self) -> &[IntervalRootCertificate] {
        &self.window.roots
            [self.selected_root_offset..self.selected_root_offset + self.selected_root_count]
    }
}

/// Canonical digest used to bind a certificate to the exact retained MPFR
/// secular weights that produced it.
pub fn production_ccm_source_weights_digest(
    weights: &[Float],
    precision_bits: u32,
) -> Result<ContentDigest> {
    let serialized = weights
        .iter()
        .map(|weight| serialize_float(&Float::with_val(precision_bits, weight), precision_bits))
        .collect::<Vec<_>>();
    weights_digest(&serialized)
}

/// Validate the identity, source binding, target accounting, and isolated-root
/// structure of a persisted independent CCM certificate without repeating the
/// expensive FLINT root census. This is the cache-read validation boundary;
/// [`verify_production_independent_ccm_root_certificate`] remains the explicit
/// full numerical replay operation.
pub fn validate_production_independent_ccm_root_certificate_structure(
    certificate: &ProductionIndependentCcmRootCertificate,
) -> Result<()> {
    if certificate.schema_version != 1
        || certificate.integer_cutoff_c <= 1
        || certificate.modes == 0
        || certificate.precision_bits <= 64
        || certificate.isolation_bits < 16
        || certificate.isolation_bits >= certificate.precision_bits
        || certificate.reference_seeds_used
        || certificate.source_weights.len() != 2 * certificate.modes + 1
        || !certificate.source_weights_digest.validate()
        || certificate.source_weights_digest != weights_digest(&certificate.source_weights)?
        || certificate.source_certification_scope
            != FiniteSourceCertificationScope::ExactStoredPointSource
        || certificate.discovery_method
            != "exact_cumulative_finite_source_counts_then_flint_arb_isolation_and_interval_newton"
        || certificate.count_method
            != "flint_arb_complete_complex_root_isolation_with_real_window_classification"
        || certificate.selected_root_count == 0
        || !certificate.window.reconciliation.complete
        || certificate.window.count.certified_root_count != certificate.window.roots.len()
        || certificate.window.reconciliation.isolated_root_count != certificate.window.roots.len()
        || certificate
            .window
            .reconciliation
            .independently_counted_roots
            != certificate.window.roots.len()
        || !certificate.window.reconciliation.ordered_and_disjoint
    {
        bail!("production independent CCM certificate structure is invalid");
    }
    certificate
        .interval_newton
        .validate(certificate.precision_bits)
        .map_err(anyhow::Error::from)?;
    let selected_end = certificate
        .selected_root_offset
        .checked_add(certificate.selected_root_count)
        .context("independent CCM selected certificate range overflows")?;
    if selected_end > certificate.window.roots.len() {
        bail!("independent CCM selected certificate range leaves its certified window");
    }
    let expected_indices = match &certificate.target {
        IndependentCcmRootTarget::Prefix { count } => {
            if *count != certificate.selected_root_count {
                bail!("independent CCM prefix certificate count is inconsistent");
            }
            Some((1, *count))
        }
        IndependentCcmRootTarget::IndexRange { first, last } => {
            if *first == 0 || first > last || *last - *first + 1 != certificate.selected_root_count
            {
                bail!("independent CCM indexed certificate target is inconsistent");
            }
            Some((*first, *last))
        }
        IndependentCcmRootTarget::PositiveHeightWindow { lower, upper } => {
            let lower = parse_endpoint(lower, certificate.precision_bits)?;
            let upper = parse_endpoint(upper, certificate.precision_bits)?;
            if lower <= 0 || lower >= upper {
                bail!("independent CCM height certificate target is invalid");
            }
            match certificate.first_selected_positive_index {
                Some(first) => Some((
                    first,
                    first
                        .checked_add(certificate.selected_root_count - 1)
                        .context("independent CCM height certificate ordinal range overflows")?,
                )),
                None => None,
            }
        }
    };
    if expected_indices
        != certificate
            .first_selected_positive_index
            .zip(certificate.last_selected_positive_index)
    {
        bail!("independent CCM certificate ordinal assignment is inconsistent");
    }
    let lower_bound = Rational::from(
        Rational::parse(&certificate.lower_bound)
            .context("parse independent CCM exact lower discovery bound")?,
    );
    let upper_bound = Rational::from(
        Rational::parse(&certificate.upper_bound)
            .context("parse independent CCM exact upper discovery bound")?,
    );
    if lower_bound < 0 || lower_bound >= upper_bound {
        bail!("independent CCM certificate discovery bounds are invalid");
    }
    for root in &certificate.window.roots {
        if root.precision_bits != certificate.precision_bits
            || root.status != IntervalRootStatus::CertifiedUnique
            || !root.uniqueness_witnessed
        {
            bail!("independent CCM certificate contains a non-unique root enclosure");
        }
        let lower = parse_endpoint(&root.lower, root.precision_bits)?;
        let upper = parse_endpoint(&root.upper, root.precision_bits)?;
        let lower_exact = lower
            .to_rational()
            .context("convert certified CCM root lower endpoint to an exact rational")?;
        let upper_exact = upper
            .to_rational()
            .context("convert certified CCM root upper endpoint to an exact rational")?;
        if lower_exact <= lower_bound || lower_exact >= upper_exact || upper_exact >= upper_bound {
            bail!("independent CCM root enclosure leaves its certified bounds");
        }
    }
    for pair in certificate.window.roots.windows(2) {
        let precision = pair[0].precision_bits.max(pair[1].precision_bits);
        if parse_endpoint(&pair[0].upper, precision)? >= parse_endpoint(&pair[1].lower, precision)?
        {
            bail!("independent CCM root enclosures are not strictly ordered and disjoint");
        }
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceZeroDataset {
    pub schema_version: u32,
    pub dataset_id: String,
    pub source_citation: String,
    pub source_revision: String,
    pub precision_bits: u32,
    pub positive_ordinates: Vec<String>,
    pub dataset_digest: ContentDigest,
}

#[derive(Serialize)]
struct ReferenceZeroDatasetEnvelope<'a> {
    schema_version: u32,
    dataset_id: &'a str,
    source_citation: &'a str,
    source_revision: &'a str,
    precision_bits: u32,
    positive_ordinates: &'a [String],
}

impl ReferenceZeroDataset {
    pub fn new(
        dataset_id: impl Into<String>,
        source_citation: impl Into<String>,
        source_revision: impl Into<String>,
        precision_bits: u32,
        positive_ordinates: Vec<String>,
    ) -> Result<Self> {
        let mut dataset = Self {
            schema_version: 1,
            dataset_id: dataset_id.into(),
            source_citation: source_citation.into(),
            source_revision: source_revision.into(),
            precision_bits,
            positive_ordinates,
            dataset_digest: ContentDigest::sha256(b"pending-reference-zero-dataset"),
        };
        dataset.dataset_digest = dataset.recompute_digest()?;
        dataset.validate()?;
        Ok(dataset)
    }

    fn recompute_digest(&self) -> Result<ContentDigest> {
        let envelope = ReferenceZeroDatasetEnvelope {
            schema_version: self.schema_version,
            dataset_id: &self.dataset_id,
            source_citation: &self.source_citation,
            source_revision: &self.source_revision,
            precision_bits: self.precision_bits,
            positive_ordinates: &self.positive_ordinates,
        };
        Ok(ContentDigest::sha256(&serde_json::to_vec(&envelope)?))
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != 1
            || self.dataset_id.trim().is_empty()
            || self.source_citation.trim().is_empty()
            || self.source_revision.trim().is_empty()
            || self.precision_bits <= 64
            || self.positive_ordinates.is_empty()
            || !self.dataset_digest.validate()
            || self.dataset_digest != self.recompute_digest()?
        {
            bail!("reference-zero dataset identity or provenance is invalid");
        }
        let values = self
            .positive_ordinates
            .iter()
            .map(|value| parse_endpoint(value, self.precision_bits))
            .collect::<Result<Vec<_>>>()?;
        if values.iter().any(|value| value <= &0)
            || values.windows(2).any(|pair| pair[0] >= pair[1])
        {
            bail!("reference-zero ordinates must be positive and strictly ordered");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RootReferenceComparisonRecord {
    pub positive_index: usize,
    pub enclosure_lower: String,
    pub enclosure_upper: String,
    pub reference_ordinate: String,
    pub absolute_midpoint_error: String,
    pub measured_agreement_digits: u32,
    pub reference_inside_enclosure: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PostDiscoveryReferenceComparison {
    pub schema_version: u32,
    pub root_certificate_digest: ContentDigest,
    pub source_weights_digest: ContentDigest,
    pub reference_dataset_digest: ContentDigest,
    pub reference_influenced_discovery: bool,
    pub records: Vec<RootReferenceComparisonRecord>,
    pub finite_comparison_statement: String,
}

fn weights_digest(weights: &[String]) -> Result<ContentDigest> {
    Ok(ContentDigest::sha256(&serde_json::to_vec(weights)?))
}

fn serialize_float(value: &Float, precision_bits: u32) -> String {
    let digits = ((precision_bits as f64) * std::f64::consts::LOG10_2)
        .ceil()
        .max(32.0) as usize
        + 2;
    value.to_string_radix(10, Some(digits))
}

fn parse_source_weights(certificate: &FirstPositiveCcmRootCertificate) -> Result<Vec<Float>> {
    certificate
        .source_weights
        .iter()
        .map(|weight| {
            let parsed = Float::parse(weight).context("parse stored CCM source weight")?;
            Ok(Float::with_val(certificate.precision_bits, parsed))
        })
        .collect()
}

/// Discover and certify the first positive roots from one finite CCM secular
/// source. Pole geometry supplies every bracket; the count is obtained
/// independently from same-sign residue monotonicity.
pub fn certify_first_positive_ccm_roots(
    weights: &[Float],
    options: &FirstPositiveCcmRootOptions,
) -> Result<FirstPositiveCcmRootCertificate> {
    if options.requested_roots == 0
        || options.requested_roots > options.modes
        || weights.len() != 2 * options.modes + 1
        || options.precision_bits <= 64
        || options.pole_interval_bisection_steps == 0
    {
        bail!("first-positive CCM root request has invalid count, source dimension, precision, or bisection policy");
    }
    let source = CertifiedSecularFunction::from_integer_ccm_state(
        options.integer_cutoff_c,
        options.modes,
        weights,
        options.precision_bits,
    )?;
    let first_pole = options.modes;
    let last_pole = options.modes + options.requested_roots;
    let independent_count = source.monotone_count_between_poles(first_pole, last_pole)?;
    let roots = (first_pole..last_pole)
        .map(|left_pole| {
            source.certify_pole_interval(
                left_pole,
                options.pole_interval_bisection_steps,
                &options.interval_newton,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let reconciliation = reconcile_complete_window(&roots, &independent_count)?;
    if !reconciliation.complete {
        bail!(
            "first-positive CCM root window is incomplete: {}",
            reconciliation.reason
        );
    }
    if roots.iter().any(|root| {
        parse_endpoint(&root.lower, root.precision_bits)
            .map(|lower| lower <= 0)
            .unwrap_or(true)
    }) {
        bail!("first-positive CCM certificate contains a nonpositive enclosure");
    }
    let source_weights = weights
        .iter()
        .map(|weight| {
            serialize_float(
                &Float::with_val(options.precision_bits, weight),
                options.precision_bits,
            )
        })
        .collect::<Vec<_>>();
    Ok(FirstPositiveCcmRootCertificate {
        schema_version: 1,
        integer_cutoff_c: options.integer_cutoff_c,
        modes: options.modes,
        requested_roots: options.requested_roots,
        precision_bits: options.precision_bits,
        source_weights_digest: weights_digest(&source_weights)?,
        source_weights,
        roots,
        independent_count,
        reconciliation,
        reference_seeds_used: false,
        discovery_method: "adjacent_ccm_pole_geometry_plus_interval_newton".to_owned(),
        count_method: "same_sign_residue_monotonicity_independent_of_isolated_candidates"
            .to_owned(),
        finite_model_statement: "certified roots of one finite CCM secular source; no claim of exact equality with limiting zeta zeros"
            .to_owned(),
    })
}

pub fn verify_first_positive_ccm_root_certificate(
    certificate: &FirstPositiveCcmRootCertificate,
) -> Result<()> {
    if certificate.schema_version != 1
        || certificate.integer_cutoff_c <= 1
        || certificate.precision_bits <= 64
        || certificate.requested_roots == 0
        || certificate.requested_roots > certificate.modes
        || certificate.roots.len() != certificate.requested_roots
        || certificate.reference_seeds_used
        || certificate.source_weights.len() != 2 * certificate.modes + 1
        || !certificate.source_weights_digest.validate()
        || certificate.source_weights_digest != weights_digest(&certificate.source_weights)?
        || certificate.independent_count.certified_root_count != certificate.requested_roots
        || certificate.independent_count.open_intervals_counted != certificate.requested_roots
        || certificate.independent_count.pole_count != certificate.requested_roots + 1
        || !certificate.independent_count.square_free
        || certificate.discovery_method != "adjacent_ccm_pole_geometry_plus_interval_newton"
        || certificate.count_method
            != "same_sign_residue_monotonicity_independent_of_isolated_candidates"
        || certificate.finite_model_statement
            != "certified roots of one finite CCM secular source; no claim of exact equality with limiting zeta zeros"
    {
        bail!("first-positive CCM root certificate identity or provenance is invalid");
    }
    let weights = parse_source_weights(certificate)?;
    let source = CertifiedSecularFunction::from_integer_ccm_state(
        certificate.integer_cutoff_c,
        certificate.modes,
        &weights,
        certificate.precision_bits,
    )?;
    let replayed_count = source.monotone_count_between_poles(
        certificate.modes,
        certificate.modes + certificate.requested_roots,
    )?;
    if replayed_count != certificate.independent_count {
        bail!("first-positive CCM root certificate count does not replay from its source");
    }
    for (offset, root) in certificate.roots.iter().enumerate() {
        if root.precision_bits != certificate.precision_bits
            || root.status != IntervalRootStatus::CertifiedUnique
            || !root.uniqueness_witnessed
        {
            bail!("first-positive CCM root has invalid precision or uniqueness status");
        }
        let enclosure = MpfrInterval::new(
            parse_endpoint(&root.lower, root.precision_bits)?,
            parse_endpoint(&root.upper, root.precision_bits)?,
        )?;
        let left_pole = &source.poles()[certificate.modes + offset];
        let right_pole = &source.poles()[certificate.modes + offset + 1];
        if enclosure.lower() <= left_pole.upper() || enclosure.upper() >= right_pole.lower() {
            bail!("first-positive CCM root enclosure leaves its assigned pole interval");
        }
        let derivative = source
            .derivative_interval(&enclosure)
            .map_err(anyhow::Error::from)?;
        if derivative.contains_zero() {
            bail!("first-positive CCM root derivative enclosure contains zero");
        }
        let midpoint = enclosure.midpoint_point();
        let midpoint_value = source
            .evaluate_interval(&midpoint)
            .map_err(anyhow::Error::from)?;
        let newton_image = midpoint.sub(
            &midpoint_value
                .div(&derivative)
                .context("replay stored CCM root interval Newton image")?,
        );
        if !newton_image.is_interior_subset_of(&enclosure) {
            bail!("first-positive CCM root uniqueness enclosure does not replay");
        }
    }
    let replay = reconcile_complete_window(&certificate.roots, &certificate.independent_count)?;
    if !replay.complete || replay != certificate.reconciliation {
        bail!("first-positive CCM root certificate fails order/count reconciliation");
    }
    if certificate.roots.iter().any(|root| {
        parse_endpoint(&root.lower, root.precision_bits)
            .map(|lower| lower <= 0)
            .unwrap_or(true)
    }) {
        bail!("first-positive CCM root certificate contains a nonpositive root");
    }
    Ok(())
}

/// Discover the smallest positive-pole window containing exactly the requested
/// number of roots of a production mixed-sign CCM secular source. The boundary
/// search and root balls use only the exact finite numerator; reference zeros
/// are not accepted by this API.
#[cfg(feature = "arb")]
pub fn certify_production_first_positive_ccm_roots(
    weights: &[Float],
    integer_cutoff_c: u64,
    modes: usize,
    requested_roots: usize,
    precision_bits: u32,
    isolation_bits: u32,
    interval_newton: &IntervalNewtonOptions,
) -> Result<ProductionFirstPositiveCcmRootCertificate> {
    if integer_cutoff_c <= 1
        || modes == 0
        || requested_roots == 0
        || requested_roots > modes
        || weights.len() != 2 * modes + 1
        || precision_bits <= 64
    {
        bail!("production first-positive CCM request is invalid");
    }
    let source = CertifiedSecularFunction::from_integer_ccm_state(
        integer_cutoff_c,
        modes,
        weights,
        precision_bits,
    )?;
    let first_pole = modes;
    let maximum_pole = 2 * modes;
    let mut low = first_pole;
    let mut high = (first_pole + requested_roots).min(maximum_pole);
    let step = (requested_roots / 5).max(4);
    loop {
        let count = source.exact_flint_numerator_count_between_poles(first_pole, high)?;
        if count.certified_root_count >= requested_roots {
            break;
        }
        low = high;
        if high == maximum_pole {
            bail!("production CCM positive pole range does not contain the requested roots");
        }
        high = (high + step).min(maximum_pole);
    }
    while high - low > 1 {
        let middle = low + (high - low) / 2;
        let count = source.exact_flint_numerator_count_between_poles(first_pole, middle)?;
        if count.certified_root_count >= requested_roots {
            high = middle;
        } else {
            low = middle;
        }
    }
    let last_pole = high;
    let preceding_window_count =
        source.exact_flint_numerator_count_between_poles(first_pole, last_pole - 1)?;
    let exact_count = source.exact_flint_numerator_count_between_poles(first_pole, last_pole)?;
    if preceding_window_count.certified_root_count >= requested_roots
        || exact_count.certified_root_count != requested_roots
        || last_pole >= maximum_pole
    {
        bail!(
            "smallest production CCM pole boundary does not isolate exactly the requested prefix"
        );
    }
    let following_window_count =
        source.exact_flint_numerator_count_between_poles(first_pole, last_pole + 1)?;
    let window = source.certify_flint_numerator_window(
        first_pole,
        last_pole,
        isolation_bits,
        interval_newton,
    )?;
    if window.count != exact_count
        || window.roots.len() != requested_roots
        || !window.reconciliation.complete
    {
        bail!("production CCM isolated window disagrees with its exact prefix count");
    }
    let source_weights = weights
        .iter()
        .map(|weight| serialize_float(&Float::with_val(precision_bits, weight), precision_bits))
        .collect::<Vec<_>>();
    Ok(ProductionFirstPositiveCcmRootCertificate {
        schema_version: 1,
        integer_cutoff_c,
        modes,
        requested_roots,
        precision_bits,
        isolation_bits,
        interval_newton: interval_newton.clone(),
        first_pole,
        last_pole,
        source_weights_digest: weights_digest(&source_weights)?,
        source_weights,
        window,
        preceding_window_count,
        following_window_count,
        reference_seeds_used: false,
        discovery_method: "exact_finite_numerator_arb_root_balls_then_interval_newton".to_owned(),
        count_method: "flint_arb_complete_complex_root_isolation_with_real_window_classification"
            .to_owned(),
        finite_model_statement: "certified first positive roots of one finite production CCM secular source; comparison with limiting zeta zeros is post-discovery and non-certifying"
            .to_owned(),
    })
}

#[cfg(feature = "arb")]
pub fn verify_production_first_positive_ccm_root_certificate(
    certificate: &ProductionFirstPositiveCcmRootCertificate,
) -> Result<()> {
    if certificate.schema_version != 1
        || certificate.reference_seeds_used
        || certificate.first_pole != certificate.modes
        || certificate.source_weights.len() != 2 * certificate.modes + 1
        || !certificate.source_weights_digest.validate()
        || certificate.source_weights_digest != weights_digest(&certificate.source_weights)?
        || certificate.discovery_method
            != "exact_finite_numerator_arb_root_balls_then_interval_newton"
        || certificate.count_method
            != "flint_arb_complete_complex_root_isolation_with_real_window_classification"
    {
        bail!("production first-positive CCM certificate identity is invalid");
    }
    let weights = certificate
        .source_weights
        .iter()
        .map(|weight| {
            Ok(Float::with_val(
                certificate.precision_bits,
                Float::parse(weight).context("parse production CCM source weight")?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let replay = certify_production_first_positive_ccm_roots(
        &weights,
        certificate.integer_cutoff_c,
        certificate.modes,
        certificate.requested_roots,
        certificate.precision_bits,
        certificate.isolation_bits,
        &certificate.interval_newton,
    )?;
    if &replay != certificate {
        bail!("production first-positive CCM certificate does not replay from its source");
    }
    Ok(())
}

#[cfg(feature = "arb")]
fn parse_discovery_boundary(text: &str, precision_bits: u32) -> Result<Rational> {
    let parsed = Float::parse(text).context("parse independent CCM discovery boundary")?;
    Float::with_val(precision_bits, parsed)
        .to_rational()
        .context("convert independent CCM discovery boundary to an exact rational")
}

#[cfg(feature = "arb")]
fn cumulative_positive_count(
    source: &CertifiedSecularFunction,
    zero_pole: usize,
    boundary_pole: usize,
) -> Result<usize> {
    if boundary_pole == zero_pole {
        return Ok(0);
    }
    Ok(source
        .exact_flint_numerator_count_between_poles(zero_pole, boundary_pole)?
        .certified_root_count)
}

/// Discover an independently indexed positive root prefix, index range, or
/// height window. The only inputs are the finite CCM source and the requested
/// target. Exact cumulative finite-source counts assign ordinals before any
/// optional comparison with an external zero dataset.
#[cfg(feature = "arb")]
pub fn certify_production_independent_ccm_roots(
    weights: &[Float],
    integer_cutoff_c: u64,
    modes: usize,
    target: &IndependentCcmRootTarget,
    precision_bits: u32,
    isolation_bits: u32,
    interval_newton: &IntervalNewtonOptions,
) -> Result<ProductionIndependentCcmRootCertificate> {
    if integer_cutoff_c <= 1
        || modes == 0
        || weights.len() != 2 * modes + 1
        || precision_bits <= 64
        || isolation_bits < 16
        || isolation_bits >= precision_bits
    {
        bail!("production independent CCM request is invalid");
    }
    let source = CertifiedSecularFunction::from_integer_ccm_state(
        integer_cutoff_c,
        modes,
        weights,
        precision_bits,
    )?;
    let (poles, _) = source.exact_numerator_data()?;
    let zero_pole = modes;
    let maximum_pole = 2 * modes;

    let (lower_bound, upper_bound, roots_before, selected_offset, selected_count, first_index) =
        match target {
            IndependentCcmRootTarget::Prefix { count } => {
                if *count == 0 {
                    bail!("independent CCM prefix count must be positive");
                }
                let requested = IndependentCcmRootTarget::IndexRange {
                    first: 1,
                    last: *count,
                };
                let certificate = certify_production_independent_ccm_roots(
                    weights,
                    integer_cutoff_c,
                    modes,
                    &requested,
                    precision_bits,
                    isolation_bits,
                    interval_newton,
                )?;
                return Ok(ProductionIndependentCcmRootCertificate {
                    target: target.clone(),
                    ..certificate
                });
            }
            IndependentCcmRootTarget::IndexRange { first, last } => {
                if *first == 0 || first > last {
                    bail!("independent CCM positive indices require 1 <= first <= last");
                }
                let available = cumulative_positive_count(&source, zero_pole, maximum_pole)?;
                if *last > available {
                    bail!(
                        "finite CCM source has only {available} positive roots inside its positive pole range; requested index {last}"
                    );
                }

                // Largest pole boundary with a cumulative count below `first`.
                let mut low = zero_pole;
                let mut high = maximum_pole;
                while high - low > 1 {
                    let middle = low + (high - low) / 2;
                    if cumulative_positive_count(&source, zero_pole, middle)? < *first {
                        low = middle;
                    } else {
                        high = middle;
                    }
                }
                let first_boundary = low;
                let before = cumulative_positive_count(&source, zero_pole, first_boundary)?;

                // Smallest pole boundary whose cumulative count reaches `last`.
                let mut low = first_boundary;
                let mut high = maximum_pole;
                while high - low > 1 {
                    let middle = low + (high - low) / 2;
                    if cumulative_positive_count(&source, zero_pole, middle)? >= *last {
                        high = middle;
                    } else {
                        low = middle;
                    }
                }
                let last_boundary = high;
                (
                    poles[first_boundary].clone(),
                    poles[last_boundary].clone(),
                    before,
                    *first - 1 - before,
                    *last - *first + 1,
                    Some(*first),
                )
            }
            IndependentCcmRootTarget::PositiveHeightWindow { lower, upper } => {
                let lower = parse_discovery_boundary(lower, precision_bits)?;
                let upper = parse_discovery_boundary(upper, precision_bits)?;
                if lower <= 0 || lower >= upper || upper > poles[maximum_pole] {
                    bail!(
                        "positive CCM height window requires 0 < lower < upper <= largest positive pole"
                    );
                }
                let before = source
                    .exact_flint_numerator_count_in_window(&poles[zero_pole], &lower)?
                    .certified_root_count;
                let selected = source
                    .exact_flint_numerator_count_in_window(&lower, &upper)?
                    .certified_root_count;
                if selected == 0 {
                    bail!("independent CCM height window contains no finite-source roots");
                }
                (lower, upper, before, 0, selected, Some(before + 1))
            }
        };

    let window = source.certify_flint_numerator_rational_window(
        &lower_bound,
        &upper_bound,
        isolation_bits,
        interval_newton,
    )?;
    if selected_offset
        .checked_add(selected_count)
        .is_none_or(|end| end > window.roots.len())
    {
        bail!("independent CCM target is not contained in its certified root window");
    }
    let last_index = first_index.and_then(|first| first.checked_add(selected_count - 1));
    let source_weights = weights
        .iter()
        .map(|weight| serialize_float(&Float::with_val(precision_bits, weight), precision_bits))
        .collect::<Vec<_>>();
    Ok(ProductionIndependentCcmRootCertificate {
        schema_version: 1,
        integer_cutoff_c,
        modes,
        precision_bits,
        isolation_bits,
        interval_newton: interval_newton.clone(),
        target: target.clone(),
        lower_bound: lower_bound.to_string(),
        upper_bound: upper_bound.to_string(),
        positive_roots_before_window: roots_before,
        selected_root_offset: selected_offset,
        selected_root_count: selected_count,
        first_selected_positive_index: first_index,
        last_selected_positive_index: last_index,
        source_weights_digest: weights_digest(&source_weights)?,
        source_certification_scope: FiniteSourceCertificationScope::ExactStoredPointSource,
        source_weights,
        window,
        reference_seeds_used: false,
        discovery_method:
            "exact_cumulative_finite_source_counts_then_flint_arb_isolation_and_interval_newton"
                .to_owned(),
        count_method:
            "flint_arb_complete_complex_root_isolation_with_real_window_classification"
                .to_owned(),
        finite_model_statement: "certified independently indexed roots of one finite production CCM secular source; external zeta-zero data did not seed, bound, count, or alter discovery"
            .to_owned(),
    })
}

#[cfg(feature = "arb")]
pub fn verify_production_independent_ccm_root_certificate(
    certificate: &ProductionIndependentCcmRootCertificate,
) -> Result<()> {
    if certificate.schema_version != 1
        || certificate.reference_seeds_used
        || certificate.source_weights.len() != 2 * certificate.modes + 1
        || !certificate.source_weights_digest.validate()
        || certificate.source_weights_digest != weights_digest(&certificate.source_weights)?
        || certificate.source_certification_scope
            != FiniteSourceCertificationScope::ExactStoredPointSource
        || certificate.discovery_method
            != "exact_cumulative_finite_source_counts_then_flint_arb_isolation_and_interval_newton"
        || certificate.count_method
            != "flint_arb_complete_complex_root_isolation_with_real_window_classification"
        || certificate.selected_root_count == 0
    {
        bail!("production independent CCM certificate identity is invalid");
    }
    let weights = certificate
        .source_weights
        .iter()
        .map(|weight| {
            Ok(Float::with_val(
                certificate.precision_bits,
                Float::parse(weight).context("parse independent CCM source weight")?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let replay = certify_production_independent_ccm_roots(
        &weights,
        certificate.integer_cutoff_c,
        certificate.modes,
        &certificate.target,
        certificate.precision_bits,
        certificate.isolation_bits,
        &certificate.interval_newton,
    )?;
    if &replay != certificate {
        bail!("production independent CCM certificate does not replay from its source");
    }
    Ok(())
}

fn certified_decimal_digits(root: &IntervalRootCertificate) -> Result<u32> {
    let lower = parse_endpoint(&root.lower, root.precision_bits)?;
    let upper = parse_endpoint(&root.upper, root.precision_bits)?;
    let mut width = Float::with_val(root.precision_bits, upper - lower);
    if width <= 0 {
        bail!("certified CCM root enclosure must have positive width");
    }
    let mut digits = 0_u32;
    while width <= 1 && digits < 100_000 {
        width *= 10;
        if width <= 1 {
            digits += 1;
        }
    }
    Ok(digits)
}

/// Convert a verified finite-root certificate into the fixed publication dataset
/// row. Accuracy is measured from certified enclosure widths, never from a
/// post-discovery reference comparison.
pub fn first_positive_certificate_convergence_row(
    sequence_index: u64,
    certificate: &FirstPositiveCcmRootCertificate,
) -> Result<ConvergenceTableRow> {
    if sequence_index == 0 {
        bail!("CCM convergence sequence indices are one-based");
    }
    verify_first_positive_ccm_root_certificate(certificate)?;
    let digits = certificate
        .roots
        .iter()
        .map(certified_decimal_digits)
        .collect::<Result<Vec<_>>>()?;
    let minimum = digits.iter().copied().min().unwrap_or(0);
    let mut ordered = digits.clone();
    ordered.sort_unstable();
    let middle = ordered.len() / 2;
    let median = if ordered.len() % 2 == 1 {
        ordered[middle].to_string()
    } else {
        let sum = u64::from(ordered[middle - 1]) + u64::from(ordered[middle]);
        if sum % 2 == 0 {
            (sum / 2).to_string()
        } else {
            format!("{}.5", sum / 2)
        }
    };
    let index_penalty = i64::from(digits[0]) - i64::from(digits[digits.len() - 1]);
    Ok(ConvergenceTableRow {
        sequence_index,
        lambda_squared: certificate.integer_cutoff_c.to_string(),
        n_modes: certificate.modes as u64,
        precision_bits: u64::from(certificate.precision_bits),
        root_count: certificate.requested_roots as u64,
        minimum_accuracy_digits: minimum.to_string(),
        median_accuracy_digits: median,
        index_penalty_digits: index_penalty.to_string(),
        completion_status: "certified".to_owned(),
    })
}

fn first_positive_certificate_digest(
    certificate: &FirstPositiveCcmRootCertificate,
) -> Result<ContentDigest> {
    verify_first_positive_ccm_root_certificate(certificate)?;
    Ok(ContentDigest::sha256(&serde_json::to_vec(certificate)?))
}

fn measured_agreement_digits(error: &Float, precision_bits: u32) -> u32 {
    if error == &0 {
        return ((precision_bits as f64) * std::f64::consts::LOG10_2).floor() as u32;
    }
    let mut scaled = Float::with_val(precision_bits, error);
    let mut digits = 0_u32;
    while scaled <= 1 && digits < 100_000 {
        scaled *= 10;
        if scaled <= 1 {
            digits += 1;
        }
    }
    digits
}

/// Compare only after reference-free discovery has produced and verified a
/// complete certificate. The reference dataset cannot alter root order,
/// enclosures, counts, or discovery provenance.
pub fn compare_first_positive_roots_to_references(
    certificate: &FirstPositiveCcmRootCertificate,
    references: &ReferenceZeroDataset,
) -> Result<PostDiscoveryReferenceComparison> {
    verify_first_positive_ccm_root_certificate(certificate)?;
    references.validate()?;
    if references.positive_ordinates.len() < certificate.requested_roots {
        bail!("reference-zero dataset does not cover the certified root window");
    }
    let precision = certificate.precision_bits.max(references.precision_bits);
    let records = certificate
        .roots
        .iter()
        .zip(&references.positive_ordinates)
        .enumerate()
        .map(|(offset, (root, reference))| {
            let lower = parse_endpoint(&root.lower, precision)?;
            let upper = parse_endpoint(&root.upper, precision)?;
            let reference_value = parse_endpoint(reference, precision)?;
            let mut midpoint = Float::with_val(precision, &lower + &upper);
            midpoint /= 2;
            let mut error = Float::with_val(precision, midpoint - &reference_value);
            error.abs_mut();
            Ok(RootReferenceComparisonRecord {
                positive_index: offset + 1,
                enclosure_lower: root.lower.clone(),
                enclosure_upper: root.upper.clone(),
                reference_ordinate: reference.clone(),
                absolute_midpoint_error: serialize_float(&error, precision),
                measured_agreement_digits: measured_agreement_digits(&error, precision),
                reference_inside_enclosure: reference_value >= lower && reference_value <= upper,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(PostDiscoveryReferenceComparison {
        schema_version: 1,
        root_certificate_digest: first_positive_certificate_digest(certificate)?,
        source_weights_digest: certificate.source_weights_digest.clone(),
        reference_dataset_digest: references.dataset_digest.clone(),
        reference_influenced_discovery: false,
        records,
        finite_comparison_statement: "post-discovery comparison of finite CCM roots with an external reference dataset; references did not seed or alter discovery"
            .to_owned(),
    })
}

pub fn verify_post_discovery_reference_comparison(
    comparison: &PostDiscoveryReferenceComparison,
    certificate: &FirstPositiveCcmRootCertificate,
    references: &ReferenceZeroDataset,
) -> Result<()> {
    let replay = compare_first_positive_roots_to_references(certificate, references)?;
    if comparison.schema_version != 1
        || comparison.reference_influenced_discovery
        || comparison.records.len() != certificate.requested_roots
        || &replay != comparison
    {
        bail!("post-discovery reference comparison does not match replayed evidence");
    }
    Ok(())
}

/// Compare a verified production mixed-sign certificate with external
/// reference ordinates only after reference-free discovery and counting have
/// completed. The comparison cannot alter any certificate field.
#[cfg(feature = "arb")]
pub fn compare_production_first_positive_roots_to_references(
    certificate: &ProductionFirstPositiveCcmRootCertificate,
    references: &ReferenceZeroDataset,
) -> Result<PostDiscoveryReferenceComparison> {
    verify_production_first_positive_ccm_root_certificate(certificate)?;
    references.validate()?;
    if references.positive_ordinates.len() < certificate.requested_roots {
        bail!("reference-zero dataset does not cover the production root window");
    }
    let precision = certificate.precision_bits.max(references.precision_bits);
    let records = certificate
        .window
        .roots
        .iter()
        .zip(&references.positive_ordinates)
        .enumerate()
        .map(|(offset, (root, reference))| {
            let lower = parse_endpoint(&root.lower, precision)?;
            let upper = parse_endpoint(&root.upper, precision)?;
            let reference_value = parse_endpoint(reference, precision)?;
            let mut midpoint = Float::with_val(precision, &lower + &upper);
            midpoint /= 2;
            let mut error = Float::with_val(precision, midpoint - &reference_value);
            error.abs_mut();
            Ok(RootReferenceComparisonRecord {
                positive_index: offset + 1,
                enclosure_lower: root.lower.clone(),
                enclosure_upper: root.upper.clone(),
                reference_ordinate: reference.clone(),
                absolute_midpoint_error: serialize_float(&error, precision),
                measured_agreement_digits: measured_agreement_digits(&error, precision),
                reference_inside_enclosure: reference_value >= lower && reference_value <= upper,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(PostDiscoveryReferenceComparison {
        schema_version: 1,
        root_certificate_digest: ContentDigest::sha256(&serde_json::to_vec(certificate)?),
        source_weights_digest: certificate.source_weights_digest.clone(),
        reference_dataset_digest: references.dataset_digest.clone(),
        reference_influenced_discovery: false,
        records,
        finite_comparison_statement: "post-discovery comparison of a finite production CCM root prefix with an external reference dataset; references did not seed, bound, count, or alter discovery"
            .to_owned(),
    })
}

#[cfg(feature = "arb")]
pub fn verify_production_post_discovery_reference_comparison(
    comparison: &PostDiscoveryReferenceComparison,
    certificate: &ProductionFirstPositiveCcmRootCertificate,
    references: &ReferenceZeroDataset,
) -> Result<()> {
    let replay = compare_production_first_positive_roots_to_references(certificate, references)?;
    if comparison.schema_version != 1
        || comparison.reference_influenced_discovery
        || comparison.records.len() != certificate.requested_roots
        || &replay != comparison
    {
        bail!("production post-discovery comparison does not match replayed evidence");
    }
    Ok(())
}

/// Attach external reference ordinates to an independently indexed production
/// window only after its exact count and root isolation replay successfully.
#[cfg(feature = "arb")]
pub fn compare_production_independent_roots_to_references(
    certificate: &ProductionIndependentCcmRootCertificate,
    references: &ReferenceZeroDataset,
) -> Result<PostDiscoveryReferenceComparison> {
    verify_production_independent_ccm_root_certificate(certificate)?;
    references.validate()?;
    let first = certificate
        .first_selected_positive_index
        .context("independent production comparison requires positive root indices")?;
    let last = certificate
        .last_selected_positive_index
        .context("independent production comparison requires positive root indices")?;
    if references.positive_ordinates.len() < last {
        bail!("reference-zero dataset does not cover the independent production window");
    }
    let precision = certificate.precision_bits.max(references.precision_bits);
    let records = certificate
        .selected_roots()
        .iter()
        .zip(&references.positive_ordinates[first - 1..last])
        .enumerate()
        .map(|(offset, (root, reference))| {
            let lower = parse_endpoint(&root.lower, precision)?;
            let upper = parse_endpoint(&root.upper, precision)?;
            let reference_value = parse_endpoint(reference, precision)?;
            let mut midpoint = Float::with_val(precision, &lower + &upper);
            midpoint /= 2;
            let mut error = Float::with_val(precision, midpoint - &reference_value);
            error.abs_mut();
            Ok(RootReferenceComparisonRecord {
                positive_index: first + offset,
                enclosure_lower: root.lower.clone(),
                enclosure_upper: root.upper.clone(),
                reference_ordinate: reference.clone(),
                absolute_midpoint_error: serialize_float(&error, precision),
                measured_agreement_digits: measured_agreement_digits(&error, precision),
                reference_inside_enclosure: reference_value >= lower && reference_value <= upper,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(PostDiscoveryReferenceComparison {
        schema_version: 1,
        root_certificate_digest: ContentDigest::sha256(&serde_json::to_vec(certificate)?),
        source_weights_digest: certificate.source_weights_digest.clone(),
        reference_dataset_digest: references.dataset_digest.clone(),
        reference_influenced_discovery: false,
        records,
        finite_comparison_statement: "post-discovery comparison of an independently indexed finite production CCM root window; references did not seed, bound, count, or alter discovery"
            .to_owned(),
    })
}

#[cfg(feature = "arb")]
pub fn verify_production_independent_post_discovery_comparison(
    comparison: &PostDiscoveryReferenceComparison,
    certificate: &ProductionIndependentCcmRootCertificate,
    references: &ReferenceZeroDataset,
) -> Result<()> {
    let replay = compare_production_independent_roots_to_references(certificate, references)?;
    if comparison.reference_influenced_discovery || &replay != comparison {
        bail!("independent production post-discovery comparison does not replay");
    }
    Ok(())
}

fn parse_endpoint(text: &str, precision: u32) -> Result<Float> {
    let parsed = Float::parse(text).context("parse certified root endpoint")?;
    Ok(Float::with_val(precision, parsed))
}

pub fn reconcile_complete_window(
    roots: &[IntervalRootCertificate],
    count: &SecularCountCertificate,
) -> Result<RootCountReconciliation> {
    let all_unique = roots.iter().all(|root| {
        root.status == IntervalRootStatus::CertifiedUnique && root.uniqueness_witnessed
    });
    let mut ordered_and_disjoint = true;
    for pair in roots.windows(2) {
        let precision = pair[0].precision_bits.max(pair[1].precision_bits);
        let left_upper = parse_endpoint(&pair[0].upper, precision)?;
        let right_lower = parse_endpoint(&pair[1].lower, precision)?;
        if left_upper >= right_lower {
            ordered_and_disjoint = false;
            break;
        }
    }
    let counts_agree = roots.len() == count.certified_root_count;
    let complete = all_unique && ordered_and_disjoint && counts_agree;
    Ok(RootCountReconciliation {
        complete,
        isolated_root_count: roots.len(),
        independently_counted_roots: count.certified_root_count,
        ordered_and_disjoint,
        reason: if complete {
            "every monotonic pole interval has one disjoint certified unique root".to_owned()
        } else if !all_unique {
            "at least one isolated candidate lacks a uniqueness certificate".to_owned()
        } else if !ordered_and_disjoint {
            "certified root enclosures overlap or are not ordered".to_owned()
        } else {
            "isolated root count differs from the independent monotone count".to_owned()
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ccm::window::{
        build_ccm_research_sequence, verify_ccm_research_sequence, CcmResearchObservation,
        CcmResearchSequenceOutcome, CcmResearchTarget,
    };
    use xc_core::DecimalLiteral;

    fn synthetic_source(precision: u32) -> CertifiedSecularFunction {
        let poles = [-1, 0, 1]
            .into_iter()
            .map(|value| Float::with_val(precision, value))
            .collect::<Vec<_>>();
        let weights = [1, 1, 1]
            .into_iter()
            .map(|value| Float::with_val(precision, value))
            .collect::<Vec<_>>();
        CertifiedSecularFunction::from_point_data(&poles, &weights, precision).unwrap()
    }

    fn interval(precision: u32, lower: f64, upper: f64) -> MpfrInterval {
        MpfrInterval::new(
            Float::with_val(precision, lower),
            Float::with_val(precision, upper),
        )
        .unwrap()
    }

    #[test]
    fn same_sign_residues_count_roots_without_using_candidates() {
        let source = synthetic_source(192);
        let count = source.monotone_count().unwrap();
        assert_eq!(count.certified_root_count, 2);
        assert_eq!(count.open_intervals_counted, 2);
    }

    #[test]
    fn finite_entire_adapter_evaluates_derivative_and_certifies_contour_count() {
        let entire = synthetic_source(192)
            .normalized_finite_entire_function()
            .unwrap();
        // The normalized numerator is z^2 - 1/3.
        assert_eq!(entire.degree(), 2);
        assert_eq!(
            entire.evaluate_real_exact(&Rational::from((0, 1))).unwrap(),
            Rational::from((-1, 3))
        );
        let one = ComplexRational {
            real: Rational::from((1, 1)),
            imaginary: Rational::from((0, 1)),
        };
        assert_eq!(
            entire.evaluate_derivative_exact(&one).unwrap(),
            ComplexRational {
                real: Rational::from((2, 1)),
                imaginary: Rational::from((0, 1)),
            }
        );
        let rectangle = RationalContourRectangle::new(
            Rational::from((-2, 1)),
            Rational::from((2, 1)),
            Rational::from((-1, 1)),
            Rational::from((1, 1)),
        )
        .unwrap();
        let count = entire.certify_zero_count(rectangle, 20).unwrap();
        assert_eq!(count.zero_count, 2);
        assert!(count.boundary_excludes_zero && count.rigorous);
    }

    #[test]
    fn ccm_constructor_records_centered_pole_geometry() {
        let precision = 128;
        let weights = vec![Float::with_val(precision, 1); 5];
        let source =
            CertifiedSecularFunction::from_integer_ccm_state(5, 2, &weights, precision).unwrap();
        assert_eq!(source.poles().len(), 5);
        assert!(source.poles()[2].contains_zero());
        assert_eq!(
            source
                .monotone_count_between_poles(2, 4)
                .unwrap()
                .certified_root_count,
            2
        );
    }

    #[test]
    fn interval_newton_isolates_and_reconciles_complete_window() {
        let precision = 192;
        let source = synthetic_source(precision);
        let options = IntervalNewtonOptions {
            width_tolerance: DecimalLiteral::new("1e-35").unwrap(),
            maximum_iterations: 20,
        };
        let left = source.certify_pole_interval(0, 12, &options).unwrap();
        let right = source.certify_pole_interval(1, 12, &options).unwrap();
        assert_eq!(left.status, IntervalRootStatus::CertifiedUnique);
        assert_eq!(right.status, IntervalRootStatus::CertifiedUnique);
        let reconciliation =
            reconcile_complete_window(&[left, right], &source.monotone_count().unwrap()).unwrap();
        assert!(reconciliation.complete, "{}", reconciliation.reason);
    }

    #[test]
    fn pole_overlap_fails_before_interval_arithmetic() {
        let precision = 128;
        let source = synthetic_source(precision);
        let error = source
            .isolate(
                &interval(precision, -0.1, 0.1),
                &IntervalNewtonOptions {
                    width_tolerance: DecimalLiteral::new("1e-20").unwrap(),
                    maximum_iterations: 10,
                },
            )
            .unwrap_err();
        assert!(error.to_string().contains("overlaps"));
    }

    #[test]
    fn real_ccm_finite_source_certifies_positive_window_without_reference_seeds() {
        let precision = 192;
        let modes = 2;
        let params = crate::ccm::CcmParams::from_lambda_sq_integer(5, modes);
        let mut config = crate::ccm::hp::HighPrecConfig::for_decimal_digits(40);
        config.precision_bits = precision;
        config.quad_points = crate::ccm::hp::MIN_QUAD_POINTS;
        let matrix = crate::ccm::hp::weil_matrix_hp(&params, &config, true).unwrap();
        let spectrum =
            xc_numerics::eigen::dense_symmetric_eigenvalues_hp(&matrix, 2 * modes + 1, precision)
                .unwrap();
        let weights = xc_numerics::eigen::dense_symmetric_eigenvector_for_value_hp(
            &matrix,
            2 * modes + 1,
            &spectrum[0],
            precision,
            20,
        )
        .unwrap();
        let source =
            CertifiedSecularFunction::from_integer_ccm_state(5, modes, &weights, precision)
                .unwrap();
        let options = IntervalNewtonOptions {
            width_tolerance: DecimalLiteral::new("1e-30").unwrap(),
            maximum_iterations: 20,
        };
        let certificate = source
            .certify_exact_numerator_window(modes, 2 * modes, 80, &options)
            .unwrap();
        assert!(certificate.count.square_free);
        assert!(
            certificate.reconciliation.complete,
            "{}",
            certificate.reconciliation.reason
        );
        #[cfg(feature = "arb")]
        {
            let flint_certificate = source
                .certify_flint_numerator_window(modes, 2 * modes, 80, &options)
                .unwrap();
            assert_eq!(flint_certificate.roots.len(), certificate.roots.len());
            assert_eq!(
                flint_certificate.count.certified_root_count,
                certificate.count.certified_root_count
            );
            assert!(flint_certificate.count.square_free);
            assert!(flint_certificate.reconciliation.complete);
        }
    }

    #[cfg(feature = "arb")]
    #[test]
    fn independent_targets_assign_ordinals_without_reference_data() {
        let precision = 192;
        let modes = 6;
        let weights = vec![Float::with_val(precision, 1); 2 * modes + 1];
        let options = IntervalNewtonOptions {
            width_tolerance: DecimalLiteral::new("1e-30").unwrap(),
            maximum_iterations: 30,
        };
        let certificate = certify_production_independent_ccm_roots(
            &weights,
            13,
            modes,
            &IndependentCcmRootTarget::IndexRange { first: 2, last: 4 },
            precision,
            64,
            &options,
        )
        .unwrap();
        assert!(!certificate.reference_seeds_used);
        assert_eq!(certificate.first_selected_positive_index, Some(2));
        assert_eq!(certificate.last_selected_positive_index, Some(4));
        assert_eq!(certificate.selected_roots().len(), 3);
        validate_production_independent_ccm_root_certificate_structure(&certificate).unwrap();
        verify_production_independent_ccm_root_certificate(&certificate).unwrap();

        let mut wrong_target = certificate.clone();
        wrong_target.target = IndependentCcmRootTarget::IndexRange { first: 1, last: 3 };
        assert!(
            validate_production_independent_ccm_root_certificate_structure(&wrong_target).is_err()
        );

        let prefix = certify_production_independent_ccm_roots(
            &weights,
            13,
            modes,
            &IndependentCcmRootTarget::Prefix { count: 4 },
            precision,
            64,
            &options,
        )
        .unwrap();
        assert_eq!(prefix.first_selected_positive_index, Some(1));
        assert_eq!(prefix.last_selected_positive_index, Some(4));
        assert_eq!(prefix.selected_roots().len(), 4);
        validate_production_independent_ccm_root_certificate_structure(&prefix).unwrap();
    }

    #[cfg(feature = "arb")]
    #[test]
    fn production_mixed_sign_prefix_replays_before_reference_comparison() {
        let precision = 192;
        let modes = 10;
        let params = crate::ccm::CcmParams::from_lambda_sq_integer(13, modes);
        let mut config = crate::ccm::hp::HighPrecConfig::for_decimal_digits(40);
        config.precision_bits = precision;
        config.quad_points = crate::ccm::hp::MIN_QUAD_POINTS;
        config.cache_mode = xc_numerics::quadrature::CacheMode::Off;
        let result = crate::ccm::hp::build_source(&params, &config).unwrap();
        let options = IntervalNewtonOptions {
            width_tolerance: DecimalLiteral::new("1e-30").unwrap(),
            maximum_iterations: 30,
        };
        let certificate = certify_production_independent_ccm_roots(
            &result.xi,
            13,
            modes,
            &IndependentCcmRootTarget::Prefix { count: 1 },
            precision,
            64,
            &options,
        )
        .unwrap();
        verify_production_independent_ccm_root_certificate(&certificate).unwrap();
        assert_eq!(certificate.window.roots.len(), 1);
        assert_eq!(certificate.positive_roots_before_window, 0);
        let root = &certificate.selected_roots()[0];
        let mut midpoint = parse_endpoint(&root.lower, precision).unwrap();
        midpoint += parse_endpoint(&root.upper, precision).unwrap();
        midpoint /= 2;
        let references = ReferenceZeroDataset::new(
            "test-only-production-midpoint",
            "test-only finite midpoint; not an external zeta dataset",
            "test-fixture-revision",
            precision,
            vec![serialize_float(&midpoint, precision)],
        )
        .unwrap();
        let comparison =
            compare_production_independent_roots_to_references(&certificate, &references).unwrap();
        verify_production_independent_post_discovery_comparison(
            &comparison,
            &certificate,
            &references,
        )
        .unwrap();
        let mut tampered = comparison;
        tampered.reference_influenced_discovery = true;
        assert!(verify_production_independent_post_discovery_comparison(
            &tampered,
            &certificate,
            &references,
        )
        .is_err());
    }

    #[test]
    fn first_fifty_positive_ccm_roots_and_growing_window_are_certified_without_seeds() {
        let configurations = [
            (10_usize, 128_u32, "1e-8", 5_u32),
            (25, 192, "1e-18", 14),
            (50, 256, "1e-35", 28),
        ];
        let mut certificates = Vec::new();
        let mut observations = Vec::new();
        for (offset, (modes, precision, tolerance, target_digits)) in
            configurations.into_iter().enumerate()
        {
            let weights = vec![Float::with_val(precision, 1); 2 * modes + 1];
            let options = FirstPositiveCcmRootOptions {
                integer_cutoff_c: 5,
                modes,
                requested_roots: modes,
                precision_bits: precision,
                pole_interval_bisection_steps: 16,
                interval_newton: IntervalNewtonOptions {
                    width_tolerance: DecimalLiteral::new(tolerance).unwrap(),
                    maximum_iterations: 30,
                },
            };
            let certificate = certify_first_positive_ccm_roots(&weights, &options).unwrap();
            verify_first_positive_ccm_root_certificate(&certificate).unwrap();
            let row = first_positive_certificate_convergence_row(offset as u64 + 1, &certificate)
                .unwrap();
            observations.push(CcmResearchObservation {
                target: CcmResearchTarget {
                    sequence_index: row.sequence_index,
                    lambda_squared: row.lambda_squared.clone(),
                    n_modes: row.n_modes,
                    precision_bits: row.precision_bits,
                    root_count: row.root_count,
                    uniform_accuracy_target_digits: target_digits,
                },
                measured: row,
                limiting_reason: None,
            });
            certificates.push(certificate);
        }
        let sequence = build_ccm_research_sequence(observations).unwrap();
        assert_eq!(sequence.outcome, CcmResearchSequenceOutcome::Achieved);
        verify_ccm_research_sequence(&sequence).unwrap();

        let certificate = certificates.last().unwrap();
        assert_eq!(certificate.roots.len(), 50);
        assert_eq!(certificate.independent_count.certified_root_count, 50);
        assert!(certificate.reconciliation.complete);
        assert!(!certificate.reference_seeds_used);

        let encoded = serde_json::to_vec(certificate).unwrap();
        let decoded: FirstPositiveCcmRootCertificate = serde_json::from_slice(&encoded).unwrap();
        verify_first_positive_ccm_root_certificate(&decoded).unwrap();

        let reference_ordinates = decoded
            .roots
            .iter()
            .map(|root| {
                let mut midpoint = parse_endpoint(&root.lower, decoded.precision_bits).unwrap();
                midpoint += parse_endpoint(&root.upper, decoded.precision_bits).unwrap();
                midpoint /= 2;
                serialize_float(&midpoint, decoded.precision_bits)
            })
            .collect();
        let references = ReferenceZeroDataset::new(
            "post-discovery-regression-fixture",
            "test-only midpoint comparison data; not an external zeta-zero source",
            "fixture-v1",
            decoded.precision_bits,
            reference_ordinates,
        )
        .unwrap();
        let comparison = compare_first_positive_roots_to_references(&decoded, &references).unwrap();
        assert!(!comparison.reference_influenced_discovery);
        assert!(comparison
            .records
            .iter()
            .all(|record| record.reference_inside_enclosure));
        verify_post_discovery_reference_comparison(&comparison, &decoded, &references).unwrap();

        let mut tampered_comparison = comparison;
        tampered_comparison.records[0].reference_inside_enclosure = false;
        assert!(verify_post_discovery_reference_comparison(
            &tampered_comparison,
            &decoded,
            &references,
        )
        .is_err());

        let mut tampered = decoded;
        tampered.roots.swap(0, 1);
        assert!(verify_first_positive_ccm_root_certificate(&tampered).is_err());
    }
}
