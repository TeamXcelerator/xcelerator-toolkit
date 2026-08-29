// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Independent finite rank-one realization of the CCM spectral operator.
//!
//! This implements CCM Lemma 5.4 directly.  For centered integer-frequency
//! diagonal `D`, normalized state `xi` with `<eta,xi>=1`, and
//! `eta=(1,...,1)`, the rank-one operator is
//!
//! `D' = D - |D xi><eta|`.
//!
//! It descends to the quotient by `span(xi)`.  The implementation constructs
//! that quotient matrix and obtains its spectrum by dense matrix reduction;
//! it does not evaluate or find zeros of the secular function.  Consequently
//! it can serve as the matrix-side route in an independence report while the
//! pole-aware secular solver remains the source-side route.

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::f64::consts::PI;
use std::fmt::{Display, Formatter};

#[derive(Clone, Debug, PartialEq)]
pub enum RankOneError {
    InvalidState(String),
    NonRealSpectrum { maximum_imaginary_part: f64 },
    Comparison(String),
}

impl Display for RankOneError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidState(message) => write!(f, "invalid CCM rank-one state: {message}"),
            Self::NonRealSpectrum {
                maximum_imaginary_part,
            } => write!(
                f,
                "rank-one quotient spectrum is not numerically real (max imaginary part {maximum_imaginary_part:e})"
            ),
            Self::Comparison(message) => write!(f, "CCM route comparison failed: {message}"),
        }
    }
}

impl Error for RankOneError {}

#[derive(Clone, Debug)]
pub struct FiniteRankOneOperatorF64 {
    centered_modes: usize,
    normalized_state: Vec<f64>,
    quotient_pivot: usize,
    quotient_matrix: DMatrix<f64>,
}

impl FiniteRankOneOperatorF64 {
    pub fn from_state(state: &[f64]) -> Result<Self, RankOneError> {
        if state.len() < 3 || state.len().is_multiple_of(2) {
            return Err(RankOneError::InvalidState(
                "state length must be odd and at least three".to_owned(),
            ));
        }
        if state.iter().any(|value| !value.is_finite()) {
            return Err(RankOneError::InvalidState(
                "state coefficients must be finite".to_owned(),
            ));
        }
        let sum = state.iter().sum::<f64>();
        if !sum.is_finite() || sum.abs() <= f64::EPSILON {
            return Err(RankOneError::InvalidState(
                "state must have a nonzero eta pairing".to_owned(),
            ));
        }
        let normalized_state = state.iter().map(|value| value / sum).collect::<Vec<_>>();
        let quotient_pivot = normalized_state
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.abs().total_cmp(&right.1.abs()))
            .map(|(index, _)| index)
            .expect("validated nonempty state");
        let pivot_value = normalized_state[quotient_pivot];
        if pivot_value == 0.0 {
            return Err(RankOneError::InvalidState(
                "could not choose a nonzero quotient pivot".to_owned(),
            ));
        }

        let centered_modes = state.len() / 2;
        let quotient_indices = (0..state.len())
            .filter(|index| *index != quotient_pivot)
            .collect::<Vec<_>>();
        let quotient_dimension = state.len() - 1;
        let mut quotient_matrix = DMatrix::<f64>::zeros(quotient_dimension, quotient_dimension);

        // Apply D' to every coordinate section vector with pivot coordinate
        // zero, then subtract its pivot component along xi to return to the
        // same section.  This is a quotient construction, not a secular
        // determinant evaluation.
        let centered_index = |index: usize| index as isize - centered_modes as isize;
        for (column, &source_index) in quotient_indices.iter().enumerate() {
            let source_frequency = centered_index(source_index) as f64;
            let mut image = Vec::with_capacity(state.len());
            for (row, &xi_row) in normalized_state.iter().enumerate() {
                let frequency = centered_index(row) as f64;
                let diagonal_action = if row == source_index {
                    source_frequency
                } else {
                    0.0
                };
                image.push(diagonal_action - frequency * xi_row);
            }
            let quotient_multiple = image[quotient_pivot] / pivot_value;
            for (row, &target_index) in quotient_indices.iter().enumerate() {
                quotient_matrix[(row, column)] =
                    image[target_index] - quotient_multiple * normalized_state[target_index];
            }
        }

        Ok(Self {
            centered_modes,
            normalized_state,
            quotient_pivot,
            quotient_matrix,
        })
    }

    pub fn centered_modes(&self) -> usize {
        self.centered_modes
    }

    pub fn normalized_state(&self) -> &[f64] {
        &self.normalized_state
    }

    pub fn quotient_pivot(&self) -> usize {
        self.quotient_pivot
    }

    pub fn quotient_matrix(&self) -> &DMatrix<f64> {
        &self.quotient_matrix
    }

    /// Dimensionless spectrum in the centered integer-frequency coordinate.
    pub fn spectrum(&self, imaginary_tolerance: f64) -> Result<Vec<f64>, RankOneError> {
        if !imaginary_tolerance.is_finite() || imaginary_tolerance <= 0.0 {
            return Err(RankOneError::InvalidState(
                "imaginary tolerance must be finite and positive".to_owned(),
            ));
        }
        let eigenvalues = self.quotient_matrix.clone().complex_eigenvalues();
        let maximum_imaginary_part = eigenvalues
            .iter()
            .map(|value| value.im.abs())
            .fold(0.0_f64, f64::max);
        if maximum_imaginary_part > imaginary_tolerance {
            return Err(RankOneError::NonRealSpectrum {
                maximum_imaginary_part,
            });
        }
        let mut real = eigenvalues.iter().map(|value| value.re).collect::<Vec<_>>();
        real.sort_by(f64::total_cmp);
        Ok(real)
    }

    /// Physical ordinates, scaled by `2*pi/L` for `L=log(lambda^2)`.
    pub fn spectrum_ordinates(
        &self,
        log_length: f64,
        imaginary_tolerance: f64,
    ) -> Result<Vec<f64>, RankOneError> {
        if !log_length.is_finite() || log_length <= 0.0 {
            return Err(RankOneError::InvalidState(
                "log length must be finite and positive".to_owned(),
            ));
        }
        let scale = 2.0 * PI / log_length;
        Ok(self
            .spectrum(imaginary_tolerance)?
            .into_iter()
            .map(|value| scale * value)
            .collect())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct IndependentRouteComparisonF64 {
    pub matrix_route: &'static str,
    pub source_route: &'static str,
    pub compared_roots: usize,
    pub maximum_absolute_difference: f64,
    pub tolerance: f64,
    pub accepted: bool,
    pub independence_rationale: &'static str,
}

/// Compare already computed finite-window values from the matrix and secular
/// routes.  This function does not use either list to seed or alter the other.
pub fn compare_independent_routes_f64(
    rank_one_values: &[f64],
    secular_values: &[f64],
    tolerance: f64,
) -> Result<IndependentRouteComparisonF64, RankOneError> {
    if rank_one_values.len() != secular_values.len() || rank_one_values.is_empty() {
        return Err(RankOneError::Comparison(
            "route lists must have equal nonzero length".to_owned(),
        ));
    }
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(RankOneError::Comparison(
            "comparison tolerance must be finite and positive".to_owned(),
        ));
    }
    if rank_one_values
        .iter()
        .chain(secular_values)
        .any(|value| !value.is_finite())
    {
        return Err(RankOneError::Comparison(
            "route values must be finite".to_owned(),
        ));
    }
    let maximum_absolute_difference = rank_one_values
        .iter()
        .zip(secular_values)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max);
    Ok(IndependentRouteComparisonF64 {
        matrix_route: "finite_rank_one_quotient_dense_spectrum_f64",
        source_route: "pole_aware_secular_root_finding_f64",
        compared_roots: rank_one_values.len(),
        maximum_absolute_difference,
        tolerance,
        accepted: maximum_absolute_difference <= tolerance,
        independence_rationale: "the matrix route diagonalizes the quotient action of D' while the source route independently brackets zeros of the meromorphic secular function; they share only the finite state and canonical frequency nodes",
    })
}

/// Independently executable semantic interpretation of one finite CCM value
/// list. Implementations must not receive results from either comparison
/// peer as seeds or inputs.
pub trait CcmSemanticEvaluatorF64: Send + Sync {
    fn route_id(&self) -> &'static str;
    fn independence_class(&self) -> &'static str;
    fn evaluate(&self) -> Result<Vec<f64>, RankOneError>;
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ThreeRouteSemanticComparisonF64 {
    pub claim_scope: String,
    pub matrix_route: String,
    pub source_route: String,
    pub zero_route: String,
    pub matrix_values: Vec<f64>,
    pub source_values: Vec<f64>,
    pub zero_values: Vec<f64>,
    pub compared_values: usize,
    pub matrix_source_maximum_difference: f64,
    pub matrix_zero_maximum_difference: f64,
    pub source_zero_maximum_difference: f64,
    pub tolerance: f64,
    pub accepted: bool,
    pub independence_classes: [String; 3],
}

fn maximum_pairwise_difference(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max)
}

/// Execute and compare matrix-contraction, source-side, and zero-side CCM
/// evaluators. Each route is invoked before comparison and receives no peer
/// output. Distinct route and independence-class identities are mandatory.
pub fn compare_three_semantic_evaluators_f64(
    matrix: &dyn CcmSemanticEvaluatorF64,
    source: &dyn CcmSemanticEvaluatorF64,
    zero: &dyn CcmSemanticEvaluatorF64,
    tolerance: f64,
) -> Result<ThreeRouteSemanticComparisonF64, RankOneError> {
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(RankOneError::Comparison(
            "semantic comparison tolerance must be finite and positive".to_owned(),
        ));
    }
    let routes = [matrix.route_id(), source.route_id(), zero.route_id()];
    let classes = [
        matrix.independence_class(),
        source.independence_class(),
        zero.independence_class(),
    ];
    if routes[0] == routes[1]
        || routes[0] == routes[2]
        || routes[1] == routes[2]
        || classes[0] == classes[1]
        || classes[0] == classes[2]
        || classes[1] == classes[2]
    {
        return Err(RankOneError::Comparison(
            "three-route comparison requires distinct route and independence-class identities"
                .to_owned(),
        ));
    }

    let matrix_values = matrix.evaluate()?;
    let source_values = source.evaluate()?;
    let zero_values = zero.evaluate()?;
    if matrix_values.is_empty()
        || matrix_values.len() != source_values.len()
        || matrix_values.len() != zero_values.len()
        || matrix_values
            .iter()
            .chain(&source_values)
            .chain(&zero_values)
            .any(|value| !value.is_finite())
    {
        return Err(RankOneError::Comparison(
            "semantic evaluators must return equal nonzero finite value lists".to_owned(),
        ));
    }

    let matrix_source_maximum_difference =
        maximum_pairwise_difference(&matrix_values, &source_values);
    let matrix_zero_maximum_difference = maximum_pairwise_difference(&matrix_values, &zero_values);
    let source_zero_maximum_difference = maximum_pairwise_difference(&source_values, &zero_values);
    let accepted = matrix_source_maximum_difference <= tolerance
        && matrix_zero_maximum_difference <= tolerance
        && source_zero_maximum_difference <= tolerance;
    let compared_values = matrix_values.len();
    Ok(ThreeRouteSemanticComparisonF64 {
        claim_scope: "finite_guinand_weil_dictionary_values".to_owned(),
        matrix_route: routes[0].to_owned(),
        source_route: routes[1].to_owned(),
        zero_route: routes[2].to_owned(),
        matrix_values,
        source_values,
        zero_values,
        compared_values,
        matrix_source_maximum_difference,
        matrix_zero_maximum_difference,
        source_zero_maximum_difference,
        tolerance,
        accepted,
        independence_classes: classes.map(str::to_owned),
    })
}

/// Verify a saved finite Guinand-Weil three-route comparison without running
/// any evaluator. This checks the evidence arithmetic and route separation;
/// it does not promote the f64 comparison to interval certification.
pub fn verify_three_route_semantic_comparison_f64(
    evidence: &ThreeRouteSemanticComparisonF64,
) -> xc_certify::VerificationReport {
    let mut errors = Vec::new();
    let routes = [
        evidence.matrix_route.as_str(),
        evidence.source_route.as_str(),
        evidence.zero_route.as_str(),
    ];
    let classes = evidence.independence_classes.each_ref().map(String::as_str);
    if evidence.claim_scope != "finite_guinand_weil_dictionary_values" {
        errors.push("semantic evidence has the wrong finite claim scope".to_owned());
    }
    if routes.iter().any(|route| route.trim().is_empty())
        || classes.iter().any(|class| class.trim().is_empty())
        || routes[0] == routes[1]
        || routes[0] == routes[2]
        || routes[1] == routes[2]
        || classes[0] == classes[1]
        || classes[0] == classes[2]
        || classes[1] == classes[2]
    {
        errors.push(
            "semantic routes and independence classes must be nonempty and distinct".to_owned(),
        );
    }
    if !evidence.tolerance.is_finite() || evidence.tolerance <= 0.0 {
        errors.push("semantic comparison tolerance must be finite and positive".to_owned());
    }
    if evidence.matrix_values.is_empty()
        || evidence.matrix_values.len() != evidence.source_values.len()
        || evidence.matrix_values.len() != evidence.zero_values.len()
        || evidence.compared_values != evidence.matrix_values.len()
        || evidence
            .matrix_values
            .iter()
            .chain(&evidence.source_values)
            .chain(&evidence.zero_values)
            .any(|value| !value.is_finite())
    {
        errors
            .push("semantic evidence value lists are empty, nonfinite, or inconsistent".to_owned());
    } else {
        let differences = [
            maximum_pairwise_difference(&evidence.matrix_values, &evidence.source_values),
            maximum_pairwise_difference(&evidence.matrix_values, &evidence.zero_values),
            maximum_pairwise_difference(&evidence.source_values, &evidence.zero_values),
        ];
        let recorded = [
            evidence.matrix_source_maximum_difference,
            evidence.matrix_zero_maximum_difference,
            evidence.source_zero_maximum_difference,
        ];
        if differences
            .iter()
            .zip(recorded)
            .any(|(computed, recorded)| {
                !recorded.is_finite()
                    || (computed - recorded).abs()
                        > 8.0 * f64::EPSILON * computed.abs().max(recorded.abs()).max(1.0)
            })
        {
            errors.push("semantic pairwise discrepancy evidence does not replay".to_owned());
        }
        let accepted = differences
            .iter()
            .all(|difference| *difference <= evidence.tolerance);
        if evidence.accepted != accepted || !accepted {
            errors.push("semantic evidence does not establish three-route agreement".to_owned());
        }
    }
    xc_certify::VerificationReport {
        valid: errors.is_empty(),
        checks: if errors.is_empty() {
            vec!["finite Guinand-Weil matrix/source/zero evidence replayed".to_owned()]
        } else {
            Vec::new()
        },
        warnings: vec![
            "f64 semantic agreement is Cross-Checked finite evidence, not interval certification"
                .to_owned(),
        ],
        errors,
    }
}

#[cfg(feature = "hp")]
pub mod hp {
    use super::RankOneError;
    use rug::Float;

    #[derive(Clone, Debug)]
    pub struct HpRankOneSpectrum {
        pub dimensionless_values: Vec<Float>,
        pub ordinates: Vec<Float>,
        pub quotient_dimension: usize,
        pub quotient_pivot: usize,
        pub maximum_kernel_residual: Float,
        pub maximum_metric_asymmetry: Float,
        pub maximum_operator_asymmetry: Float,
        pub method: &'static str,
    }

    fn hp_invalid(message: impl Into<String>) -> RankOneError {
        RankOneError::InvalidState(message.into())
    }

    /// HP matrix-side spectrum of the finite CCM quotient operator.
    ///
    /// `weil_matrix - smallest_eigenvalue*I` supplies the positive
    /// semidefinite metric whose radical is `state`.  On the section
    /// `ker(eta)`, the metric matrix is `G=B^T T B` and the operator form is
    /// `H=B^T T D B`.  A Cholesky congruence reduces `H v=lambda G v` to a
    /// symmetric standard eigenproblem, independently of secular root
    /// evaluation.
    pub fn spectrum_from_weil_metric(
        weil_matrix: &[Float],
        state: &[Float],
        smallest_eigenvalue: &Float,
        log_length: &Float,
        validation_tolerance: &Float,
    ) -> Result<HpRankOneSpectrum, RankOneError> {
        let dimension = state.len();
        if dimension < 3
            || dimension.is_multiple_of(2)
            || weil_matrix.len() != dimension.saturating_mul(dimension)
        {
            return Err(hp_invalid(
                "HP Weil metric requires an odd state dimension and a matching square matrix",
            ));
        }
        let precision = state[0].prec();
        if state.iter().any(|value| !value.is_finite())
            || weil_matrix.iter().any(|value| !value.is_finite())
            || !smallest_eigenvalue.is_finite()
            || !log_length.is_finite()
            || log_length <= &Float::with_val(precision, 0)
            || validation_tolerance <= &Float::with_val(precision, 0)
        {
            return Err(hp_invalid("HP rank-one inputs must be finite and valid"));
        }

        let mut state_sum = Float::with_val(precision, 0);
        for value in state {
            state_sum += value;
        }
        if state_sum.is_zero() {
            return Err(hp_invalid("HP state must have nonzero eta pairing"));
        }
        let normalized_state = state
            .iter()
            .map(|value| {
                let mut normalized = value.clone();
                normalized /= &state_sum;
                normalized
            })
            .collect::<Vec<_>>();
        let quotient_pivot = normalized_state
            .iter()
            .enumerate()
            .max_by(|left, right| {
                left.1
                    .clone()
                    .abs()
                    .partial_cmp(&right.1.clone().abs())
                    .expect("finite HP state coefficients")
            })
            .map(|(index, _)| index)
            .expect("validated nonempty state");
        let quotient_indices = (0..dimension)
            .filter(|index| *index != quotient_pivot)
            .collect::<Vec<_>>();
        let quotient_dimension = dimension - 1;

        let mut centered = weil_matrix.to_vec();
        for diagonal in 0..dimension {
            centered[diagonal * dimension + diagonal] -= smallest_eigenvalue;
        }

        let mut maximum_kernel_residual = Float::with_val(precision, 0);
        for row in 0..dimension {
            let mut residual = Float::with_val(precision, 0);
            for column in 0..dimension {
                let mut term = centered[row * dimension + column].clone();
                term *= &normalized_state[column];
                residual += term;
            }
            let absolute = residual.abs();
            if absolute > maximum_kernel_residual {
                maximum_kernel_residual = absolute;
            }
        }
        if maximum_kernel_residual > *validation_tolerance {
            return Err(hp_invalid(format!(
                "shifted Weil metric does not annihilate the supplied state (residual {})",
                maximum_kernel_residual.to_string_radix(10, Some(20))
            )));
        }

        let centered_frequency =
            |index: usize| -> isize { index as isize - (dimension / 2) as isize };
        let mut metric = vec![Float::with_val(precision, 0); quotient_dimension.pow(2)];
        let mut operator = vec![Float::with_val(precision, 0); quotient_dimension.pow(2)];
        for (row, &row_index) in quotient_indices.iter().enumerate() {
            for (column, &column_index) in quotient_indices.iter().enumerate() {
                let mut g = centered[row_index * dimension + column_index].clone();
                g -= &centered[row_index * dimension + quotient_pivot];
                g -= &centered[quotient_pivot * dimension + column_index];
                g += &centered[quotient_pivot * dimension + quotient_pivot];
                metric[row * quotient_dimension + column] = g;

                let mut first = centered[row_index * dimension + column_index].clone();
                first -= &centered[quotient_pivot * dimension + column_index];
                first *= centered_frequency(column_index);
                let mut second = centered[row_index * dimension + quotient_pivot].clone();
                second -= &centered[quotient_pivot * dimension + quotient_pivot];
                second *= centered_frequency(quotient_pivot);
                first -= second;
                operator[row * quotient_dimension + column] = first;
            }
        }

        let maximum_metric_asymmetry = symmetrize_checked(
            &mut metric,
            quotient_dimension,
            validation_tolerance,
            "quotient metric",
        )?;
        let maximum_operator_asymmetry = symmetrize_checked(
            &mut operator,
            quotient_dimension,
            validation_tolerance,
            "quotient operator form",
        )?;
        let lower = cholesky(&metric, quotient_dimension, precision)?;
        let transformed = cholesky_congruence(&lower, &operator, quotient_dimension, precision);
        let dimensionless_values = xc_numerics::eigen::dense_symmetric_eigenvalues_hp(
            &transformed,
            quotient_dimension,
            precision,
        )
        .map_err(|error| hp_invalid(format!("HP quotient eigensolve failed: {error}")))?;
        let mut scale = Float::with_val(precision, rug::float::Constant::Pi);
        scale *= 2u32;
        scale /= log_length;
        let ordinates = dimensionless_values
            .iter()
            .map(|value| {
                let mut ordinate = value.clone();
                ordinate *= &scale;
                ordinate
            })
            .collect();
        Ok(HpRankOneSpectrum {
            dimensionless_values,
            ordinates,
            quotient_dimension,
            quotient_pivot,
            maximum_kernel_residual,
            maximum_metric_asymmetry,
            maximum_operator_asymmetry,
            method: "finite_rank_one_weil_metric_cholesky_congruence_hp",
        })
    }

    fn symmetrize_checked(
        matrix: &mut [Float],
        dimension: usize,
        tolerance: &Float,
        name: &str,
    ) -> Result<Float, RankOneError> {
        let precision = tolerance.prec();
        let mut maximum = Float::with_val(precision, 0);
        for row in 0..dimension {
            for column in row + 1..dimension {
                let mut difference = matrix[row * dimension + column].clone();
                difference -= &matrix[column * dimension + row];
                difference.abs_mut();
                if difference > maximum {
                    maximum = difference;
                }
                let mut average = matrix[row * dimension + column].clone();
                average += &matrix[column * dimension + row];
                average /= 2u32;
                matrix[row * dimension + column] = average.clone();
                matrix[column * dimension + row] = average;
            }
        }
        if maximum > *tolerance {
            return Err(hp_invalid(format!(
                "{name} asymmetry {} exceeds validation tolerance",
                maximum.to_string_radix(10, Some(20))
            )));
        }
        Ok(maximum)
    }

    fn cholesky(
        matrix: &[Float],
        dimension: usize,
        precision: u32,
    ) -> Result<Vec<Float>, RankOneError> {
        let mut lower = vec![Float::with_val(precision, 0); dimension * dimension];
        for row in 0..dimension {
            for column in 0..=row {
                let mut value = matrix[row * dimension + column].clone();
                for prior in 0..column {
                    let mut term = lower[row * dimension + prior].clone();
                    term *= &lower[column * dimension + prior];
                    value -= term;
                }
                if row == column {
                    if value <= 0 {
                        return Err(hp_invalid(format!(
                            "quotient Weil metric is not positive definite at Cholesky pivot {row}"
                        )));
                    }
                    lower[row * dimension + column] = value.sqrt();
                } else {
                    value /= &lower[column * dimension + column];
                    lower[row * dimension + column] = value;
                }
            }
        }
        Ok(lower)
    }

    fn cholesky_congruence(
        lower: &[Float],
        operator: &[Float],
        dimension: usize,
        precision: u32,
    ) -> Vec<Float> {
        let mut left_solved = vec![Float::with_val(precision, 0); dimension * dimension];
        for row in 0..dimension {
            for column in 0..dimension {
                let mut value = operator[row * dimension + column].clone();
                for prior in 0..row {
                    let mut term = lower[row * dimension + prior].clone();
                    term *= &left_solved[prior * dimension + column];
                    value -= term;
                }
                value /= &lower[row * dimension + row];
                left_solved[row * dimension + column] = value;
            }
        }
        let mut transformed = vec![Float::with_val(precision, 0); dimension * dimension];
        for row in 0..dimension {
            for column in 0..dimension {
                let mut value = left_solved[row * dimension + column].clone();
                for prior in 0..column {
                    let mut term = lower[column * dimension + prior].clone();
                    term *= &transformed[row * dimension + prior];
                    value -= term;
                }
                value /= &lower[column * dimension + column];
                transformed[row * dimension + column] = value;
            }
        }
        for row in 0..dimension {
            for column in row + 1..dimension {
                let mut average = transformed[row * dimension + column].clone();
                average += &transformed[column * dimension + row];
                average /= 2u32;
                transformed[row * dimension + column] = average.clone();
                transformed[column * dimension + row] = average;
            }
        }
        transformed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ccm::window::{discover_roots_f64, DiscoveryOptionsF64, SecularFunctionF64};

    struct MatrixContractionEvaluator {
        state: Vec<f64>,
    }

    impl CcmSemanticEvaluatorF64 for MatrixContractionEvaluator {
        fn route_id(&self) -> &'static str {
            "finite_rank_one_quotient_matrix_contraction"
        }

        fn independence_class(&self) -> &'static str {
            "dense_quotient_diagonalization"
        }

        fn evaluate(&self) -> Result<Vec<f64>, RankOneError> {
            FiniteRankOneOperatorF64::from_state(&self.state)?.spectrum(1e-12)
        }
    }

    struct SourceSideEvaluator {
        state: Vec<f64>,
    }

    impl CcmSemanticEvaluatorF64 for SourceSideEvaluator {
        fn route_id(&self) -> &'static str {
            "pole_aware_source_secular_evaluation"
        }

        fn independence_class(&self) -> &'static str {
            "meromorphic_source_root_finding"
        }

        fn evaluate(&self) -> Result<Vec<f64>, RankOneError> {
            let secular = SecularFunctionF64::new(vec![-1.0, 0.0, 1.0], self.state.clone())
                .map_err(|error| {
                    RankOneError::Comparison(format!("construct source evaluator: {error}"))
                })?;
            let records =
                discover_roots_f64(&secular, -0.99, 0.99, &DiscoveryOptionsF64::default())
                    .map_err(|error| {
                        RankOneError::Comparison(format!("evaluate source route: {error}"))
                    })?;
            records
                .iter()
                .map(|record| {
                    record.midpoint.parse::<f64>().map_err(|error| {
                        RankOneError::Comparison(format!("parse source value: {error}"))
                    })
                })
                .collect()
        }
    }

    struct ZeroSideExplicitFormulaEvaluator;

    impl CcmSemanticEvaluatorF64 for ZeroSideExplicitFormulaEvaluator {
        fn route_id(&self) -> &'static str {
            "zero_side_explicit_polynomial_formula"
        }

        fn independence_class(&self) -> &'static str {
            "closed_form_zero_evaluation"
        }

        fn evaluate(&self) -> Result<Vec<f64>, RankOneError> {
            let root = 1.0 / 3.0_f64.sqrt();
            Ok(vec![-root, root])
        }
    }

    #[test]
    fn quotient_matrix_matches_secular_roots_without_using_them_as_seeds() {
        // xi=(-) is symmetric and normalized internally.  The secular
        // numerator is 3s^2-1, hence the two quotient eigenvalues are
        // +/-1/sqrt(3).
        let state = vec![1.0, 1.0, 1.0];
        let operator = FiniteRankOneOperatorF64::from_state(&state).unwrap();
        let matrix_values = operator.spectrum(1e-12).unwrap();
        assert_eq!(matrix_values.len(), 2);
        assert!((matrix_values[0] + 1.0 / 3.0_f64.sqrt()).abs() < 1e-12);
        assert!((matrix_values[1] - 1.0 / 3.0_f64.sqrt()).abs() < 1e-12);

        let secular = SecularFunctionF64::new(vec![-1.0, 0.0, 1.0], state).unwrap();
        let records =
            discover_roots_f64(&secular, -0.99, 0.99, &DiscoveryOptionsF64::default()).unwrap();
        let secular_values = records
            .iter()
            .map(|record| record.midpoint.parse::<f64>().unwrap())
            .collect::<Vec<_>>();
        let comparison =
            compare_independent_routes_f64(&matrix_values, &secular_values, 1e-10).unwrap();
        assert!(comparison.accepted);
        assert!(comparison.independence_rationale.contains("quotient"));

        let semantic = compare_three_semantic_evaluators_f64(
            &MatrixContractionEvaluator {
                state: vec![1.0, 1.0, 1.0],
            },
            &SourceSideEvaluator {
                state: vec![1.0, 1.0, 1.0],
            },
            &ZeroSideExplicitFormulaEvaluator,
            1e-10,
        )
        .unwrap();
        assert!(semantic.accepted);
        assert_eq!(semantic.compared_values, 2);
        assert_eq!(
            semantic.independence_classes,
            [
                "dense_quotient_diagonalization",
                "meromorphic_source_root_finding",
                "closed_form_zero_evaluation"
            ]
        );
        let encoded = serde_json::to_string(&semantic).unwrap();
        let decoded: ThreeRouteSemanticComparisonF64 = serde_json::from_str(&encoded).unwrap();
        let verified = verify_three_route_semantic_comparison_f64(&decoded);
        assert!(verified.valid, "{:?}", verified.errors);
        assert!(verified.warnings[0].contains("not interval certification"));

        let mut tampered = decoded;
        tampered.zero_values[0] += 0.25;
        assert!(!verify_three_route_semantic_comparison_f64(&tampered).valid);
    }

    #[test]
    fn three_route_comparison_rejects_nominally_duplicated_routes() {
        let matrix = MatrixContractionEvaluator {
            state: vec![1.0, 1.0, 1.0],
        };
        let error =
            compare_three_semantic_evaluators_f64(&matrix, &matrix, &matrix, 1e-10).unwrap_err();
        assert!(matches!(error, RankOneError::Comparison(_)));
    }

    #[test]
    fn quotient_is_invariant_under_state_scaling_and_pivot_choice_is_recorded() {
        let left = FiniteRankOneOperatorF64::from_state(&[2.0, 4.0, 2.0]).unwrap();
        let right = FiniteRankOneOperatorF64::from_state(&[1.0, 2.0, 1.0]).unwrap();
        assert_eq!(left.quotient_pivot(), 1);
        let left_spectrum = left.spectrum(1e-12).unwrap();
        let right_spectrum = right.spectrum(1e-12).unwrap();
        for (left, right) in left_spectrum.iter().zip(right_spectrum) {
            assert!((left - right).abs() < 1e-12);
        }
    }

    #[cfg(feature = "hp")]
    #[test]
    fn hp_weil_metric_route_recovers_the_same_quotient_spectrum() {
        use super::hp::spectrum_from_weil_metric;
        use rug::float::Constant;
        use rug::Float;

        let precision = 192;
        // Complete-graph Laplacian: positive semidefinite with radical
        // span(1,1,1), and off-diagonal divided-difference structure for
        // centered frequencies -1,0,1.
        let integers = [2, -1, -1, -1, 2, -1, -1, -1, 2];
        let metric = integers
            .into_iter()
            .map(|value| Float::with_val(precision, value))
            .collect::<Vec<_>>();
        let state = vec![Float::with_val(precision, 1); 3];
        let epsilon = Float::with_val(precision, 0);
        let mut log_length = Float::with_val(precision, Constant::Pi);
        log_length *= 2u32;
        let tolerance = Float::with_val(precision, Float::parse("1e-50").expect("valid tolerance"));
        let spectrum =
            spectrum_from_weil_metric(&metric, &state, &epsilon, &log_length, &tolerance).unwrap();
        assert_eq!(spectrum.dimensionless_values.len(), 2);
        let mut expected = Float::with_val(precision, 3).sqrt();
        expected.recip_mut();
        let mut left_error = spectrum.dimensionless_values[0].clone();
        left_error += &expected;
        left_error.abs_mut();
        let mut right_error = spectrum.dimensionless_values[1].clone();
        right_error -= &expected;
        right_error.abs_mut();
        assert!(left_error < tolerance);
        assert!(right_error < tolerance);
        assert!(spectrum.maximum_kernel_residual.is_zero());
    }
}
