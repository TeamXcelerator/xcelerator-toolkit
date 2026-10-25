//! Explicit approximation contracts alongside computed action residuals.
use crate::SolverError;
use serde::{Deserialize, Serialize};
use xc_operator::{ApplicationErrorBound, LinearOperator};

/// Error introduced by the operator evaluator, separate from scalar roundoff.
/// Convergence and cross-check agreement refer to the evaluated actions. This
/// record preserves their relation to the intended operator; it is not an
/// interval certificate. Missing legacy evidence remains Unknown.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OperatorErrorEvidenceF64 {
    pub approximations: xc_core::ApproximationLedger,
    pub application_error: ApplicationErrorBound<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metric_application_error: Option<ApplicationErrorBound<f64>>,
    /// Computed action residual plus declared application allowances, added
    /// outward in binary64. Excludes unbounded ordinary arithmetic roundoff.
    /// None means an unknown allowance or an unrepresentable total.
    pub residual_with_application_error: Option<f64>,
}

fn allowance(bound: &ApplicationErrorBound<f64>) -> Result<Option<f64>, SolverError> {
    match bound {
        ApplicationErrorBound::Unknown => Ok(None),
        ApplicationErrorBound::Exact => Ok(Some(0.0)),
        ApplicationErrorBound::Absolute(x) if x.is_finite() && *x >= 0.0 => Ok(Some(*x)),
        _ => Err(SolverError::InvalidConfiguration(
            "invalid operator application error bound".into(),
        )),
    }
}
fn add_up(a: f64, b: f64) -> f64 {
    if b == 0.0 {
        a
    } else if a == 0.0 {
        b
    } else {
        (a + b).next_up()
    }
}
impl OperatorErrorEvidenceF64 {
    pub(crate) fn from_operator(
        operator: &dyn LinearOperator<f64>,
        residual: f64,
    ) -> Result<Self, SolverError> {
        let application_error = operator.application_error_bound();
        let total = allowance(&application_error)?
            .map(|e| add_up(residual, e))
            .filter(|x| x.is_finite());
        let approximations = application_ledger(&application_error, None);
        Ok(Self {
            approximations,
            application_error,
            metric_application_error: None,
            residual_with_application_error: total,
        })
    }
    pub(crate) fn generalized(
        operator: &dyn LinearOperator<f64>,
        metric: &dyn LinearOperator<f64>,
        eigenvalue: f64,
        residual: f64,
    ) -> Result<Self, SolverError> {
        let mut result = Self::from_operator(operator, residual)?;
        let bound = metric.application_error_bound();
        result.residual_with_application_error =
            match (result.residual_with_application_error, allowance(&bound)?) {
                (Some(r), Some(e)) => {
                    let weighted = if e == 0.0 || eigenvalue == 0.0 {
                        0.0
                    } else {
                        (eigenvalue.abs() * e).next_up()
                    };
                    let total = add_up(r, weighted);
                    total.is_finite().then_some(total)
                }
                _ => None,
            };
        result.metric_application_error = Some(bound);
        result.approximations = application_ledger(
            &result.application_error,
            result.metric_application_error.as_ref(),
        );
        Ok(result)
    }
    pub(crate) fn validate(&self) -> Result<(), SolverError> {
        allowance(&self.application_error)?;
        self.approximations
            .validate_for_assurance(xc_core::AssuranceLevel::Computed)
            .map_err(|e| SolverError::InvalidConfiguration(e.to_string()))?;
        if let Some(bound) = &self.metric_application_error {
            allowance(bound)?;
        }
        if self
            .residual_with_application_error
            .is_some_and(|x| !x.is_finite() || x < 0.0)
        {
            return Err(SolverError::NumericalBreakdown(
                "invalid original-operator residual diagnostic".into(),
            ));
        }
        Ok(())
    }
}

/// MPFR action-error evidence for reported Ritz candidates. The maximum combines
/// their computed residuals with declared operator and metric allowances using
/// directed rounding. It excludes ordinary arithmetic error and is not a proof.
#[cfg(feature = "hp-reference")]
#[derive(Clone, Debug)]
pub struct OperatorErrorEvidenceHp {
    pub approximations: xc_core::ApproximationLedger,
    pub application_error: ApplicationErrorBound<rug::Float>,
    pub metric_application_error: Option<ApplicationErrorBound<rug::Float>>,
    /// Maximum over retained eigenpairs; separate boundary-cluster diagnostics
    /// are not included in this total.
    pub maximum_residual_with_application_error: Option<rug::Float>,
}
#[cfg(feature = "hp-reference")]
impl OperatorErrorEvidenceHp {
    pub(crate) fn for_residuals<'a>(
        operator: &dyn LinearOperator<rug::Float>,
        metric: Option<&dyn LinearOperator<rug::Float>>,
        pairs: impl Iterator<Item = (&'a rug::Float, &'a rug::Float)>,
        precision: u32,
    ) -> Result<Self, SolverError> {
        use rug::{
            float::Round,
            ops::{AddAssignRound, MulAssignRound},
            Float,
        };
        let application_error = operator.application_error_bound();
        let metric_application_error = metric.map(LinearOperator::application_error_bound);
        let allowance =
            |bound: &ApplicationErrorBound<Float>| -> Result<Option<Float>, SolverError> {
                match bound {
                    ApplicationErrorBound::Unknown => Ok(None),
                    ApplicationErrorBound::Exact => Ok(Some(Float::with_val(precision, 0))),
                    ApplicationErrorBound::Absolute(x) if x.is_finite() && x >= &0 => {
                        Ok(Some(Float::with_val_round(precision, x, Round::Up).0))
                    }
                    _ => Err(SolverError::InvalidConfiguration(
                        "invalid HP operator application error bound".into(),
                    )),
                }
            };
        let a = allowance(&application_error)?;
        let b = metric_application_error
            .as_ref()
            .map(&allowance)
            .transpose()?
            .unwrap_or_else(|| Some(Float::with_val(precision, 0)));
        let mut maximum = None;
        if let (Some(a), Some(b)) = (a, b) {
            for (lambda, residual) in pairs {
                if !lambda.is_finite() || !residual.is_finite() || residual < &0 {
                    return Err(SolverError::NumericalBreakdown(
                        "invalid HP residual evidence".into(),
                    ));
                }
                let mut total =
                    Float::with_val_round(precision, &lambda.clone().abs(), Round::Up).0;
                total.mul_assign_round(&b, Round::Up);
                total.add_assign_round(&a, Round::Up);
                total.add_assign_round(residual, Round::Up);
                if !total.is_finite() {
                    maximum = None;
                    break;
                }
                maximum = Some(maximum.map_or(total.clone(), |v: Float| v.max(&total)));
            }
        }
        let approximations =
            application_ledger(&application_error, metric_application_error.as_ref());
        Ok(Self {
            approximations,
            application_error,
            metric_application_error,
            maximum_residual_with_application_error: maximum,
        })
    }
}

fn application_ledger<S>(
    operator: &ApplicationErrorBound<S>,
    metric: Option<&ApplicationErrorBound<S>>,
) -> xc_core::ApproximationLedger {
    let entries = std::iter::once(("operator", operator)).chain(metric.map(|bound| ("metric", bound))).filter(|(_, bound)| !matches!(bound, ApplicationErrorBound::Exact)).map(|(role, _)| xc_core::ApproximationEvidence {
        kind: xc_core::ApproximationKind::OperatorApplication,
        purpose: format!("{role} action approximation participates in the computed residual; see application-error evidence; a numeric declaration alone supplies no replayable proof"),
        decisive_for_accepted_result: true,
        rigorous_bound: None,
    }).collect();
    xc_core::ApproximationLedger { entries }
}
impl Default for OperatorErrorEvidenceF64 {
    fn default() -> Self {
        Self {
            application_error: ApplicationErrorBound::Unknown,
            metric_application_error: None,
            residual_with_application_error: None,
            approximations: application_ledger(&ApplicationErrorBound::<f64>::Unknown, None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_bounds_and_invalid_allowances_do_not_become_proofs() {
        let legacy = OperatorErrorEvidenceF64::default();
        assert_eq!(legacy.residual_with_application_error, None);
        assert!(legacy
            .approximations
            .validate_for_assurance(xc_core::AssuranceLevel::Certified)
            .is_err());
        assert!(allowance(&ApplicationErrorBound::Absolute(-1.0)).is_err());
        assert!(allowance(&ApplicationErrorBound::Absolute(f64::NAN)).is_err());
        let op = xc_operator::MatrixFreeSymmetricF64::approximate(
            "bounded",
            1,
            Some(2.0),
            1.0,
            |x, y| {
                y.copy_from_slice(x);
                Ok(())
            },
        )
        .unwrap();
        let evidence = OperatorErrorEvidenceF64::generalized(&op, &op, 2.0, 0.25).unwrap();
        assert!(evidence.residual_with_application_error.unwrap() >= 3.25);
        assert_eq!(evidence.approximations.entries.len(), 2);
        assert!(evidence
            .approximations
            .validate_for_assurance(xc_core::AssuranceLevel::Certified)
            .is_err());
        assert!(
            OperatorErrorEvidenceF64::generalized(&op, &op, f64::MAX, f64::MAX)
                .unwrap()
                .residual_with_application_error
                .is_none()
        );
    }

    #[cfg(feature = "hp-reference")]
    #[test]
    fn hp_bounds_remain_outward_and_unknown_is_explicit() {
        use rug::Float;
        struct Op {
            bound: ApplicationErrorBound<Float>,
        }
        impl LinearOperator<Float> for Op {
            fn dimension(&self) -> usize {
                1
            }
            fn apply(
                &self,
                x: &[Float],
                y: &mut [Float],
            ) -> Result<(), xc_operator::OperatorError> {
                y.clone_from_slice(x);
                Ok(())
            }
            fn metadata(&self) -> xc_operator::OperatorMetadata {
                xc_operator::DenseSymmetricHp::new(
                    "fixture",
                    1,
                    vec![Float::with_val(128, 1)],
                    128,
                    &Float::with_val(128, 0),
                )
                .unwrap()
                .metadata()
            }
            fn application_error_bound(&self) -> ApplicationErrorBound<Float> {
                self.bound.clone()
            }
        }
        let op = Op {
            bound: ApplicationErrorBound::Absolute(Float::with_val(128, 1)),
        };
        let lambda = Float::with_val(128, 2);
        let residual = Float::with_val(128, 0.25);
        let evidence = OperatorErrorEvidenceHp::for_residuals(
            &op,
            Some(&op),
            std::iter::once((&lambda, &residual)),
            128,
        )
        .unwrap();
        assert!(evidence.maximum_residual_with_application_error.unwrap() >= 3.25);
        assert_eq!(evidence.approximations.entries.len(), 2);
        let unknown = Op {
            bound: ApplicationErrorBound::Unknown,
        };
        let evidence = OperatorErrorEvidenceHp::for_residuals(
            &unknown,
            None,
            std::iter::once((&lambda, &residual)),
            128,
        )
        .unwrap();
        assert!(evidence.maximum_residual_with_application_error.is_none());
        assert!(evidence
            .approximations
            .validate_for_assurance(xc_core::AssuranceLevel::Certified)
            .is_err());
    }
    #[test]
    fn crosscheck_preserves_declared_operator_approximation_bound() {
        use crate::*;
        use xc_core::*;
        use xc_operator::*;
        // Intended exact operator is the one-dimensional identity. The evaluator's
        // added error is clamp(x,-1,1), bounded by 1 for every real input.
        let op = MatrixFreeSymmetricF64::approximate(
            "identity with bounded approximate action",
            1,
            Some(2.0),
            1.0,
            |x, y| {
                y[0] = x[0] + x[0].clamp(-1.0, 1.0);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(
            op.application_error_bound(),
            ApplicationErrorBound::Absolute(1.0)
        );
        let config = SolverConfig {
            target: EigenTarget::AlgebraicLargest,
            subspace: Subspace::Full,
            assurance: AssuranceLevel::CrossChecked,
            precision: PrecisionPolicy::fixed(53),
            stopping: StoppingPolicy {
                absolute_residual: DecimalLiteral::new("1e-12").unwrap(),
                scaled_backward_error: DecimalLiteral::new("1e-12").unwrap(),
                maximum_iterations: 100,
                minimum_iterations: 2,
            },
            reproducibility: Reproducibility::Deterministic,
            algorithm_preferences: vec![],
            allow_lower_precision_seed: false,
            allow_randomized_seed: false,
        };
        let report = cross_check_f64(
            &DenseReferenceSolverF64::default(),
            &ShiftedPowerSolverF64,
            &SymmetricProblemF64::new(&op),
            &config,
            1e-12,
        )
        .unwrap();
        assert_eq!(report.accepted.eigenvalue, 2.0);
        assert_eq!(report.accepted.residual_norm, 0.0);
        assert_eq!(report.accepted.assurance, AssuranceLevel::CrossChecked);
        assert_eq!(
            report.accepted.operator_error.application_error,
            ApplicationErrorBound::Absolute(1.0)
        );
        assert!(
            report
                .accepted
                .operator_error
                .residual_with_application_error
                .unwrap()
                >= 1.0
        );
        let encoded = serde_json::to_value(&report).unwrap();
        assert_eq!(
            encoded["accepted"]["operator_error"]["application_error"]["absolute"],
            1.0
        );
        println!("approximate operator: actual eigenvalue=1, reported eigenvalue={}, reported residual={}, assurance={:?}, declared action error=1",report.accepted.eigenvalue,report.accepted.residual_norm,report.accepted.assurance);
    }
}
