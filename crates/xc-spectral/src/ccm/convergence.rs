//! Typed evidence for the three CCM convergence budgets and their sweeps.

use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CcmConvergenceError(pub String);

impl Display for CcmConvergenceError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for CcmConvergenceError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BudgetStatus {
    Satisfied,
    Unsatisfied,
    Inconclusive,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConvergenceBudgetAssessment {
    pub estimated_digits: f64,
    pub measured_digits: f64,
    pub status: BudgetStatus,
    pub diagnostic: String,
}

impl ConvergenceBudgetAssessment {
    fn validate(&self, name: &str) -> Result<(), CcmConvergenceError> {
        if !self.estimated_digits.is_finite()
            || !self.measured_digits.is_finite()
            || self.estimated_digits < 0.0
            || self.measured_digits < 0.0
            || self.diagnostic.trim().is_empty()
        {
            return Err(CcmConvergenceError(format!(
                "{name} convergence budget is incomplete or nonfinite"
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CcmConvergenceBudgetReport {
    pub arithmetic_precision: ConvergenceBudgetAssessment,
    pub mode_resolution: ConvergenceBudgetAssessment,
    pub prime_or_continuum_accuracy: ConvergenceBudgetAssessment,
}

impl CcmConvergenceBudgetReport {
    pub fn validate(&self) -> Result<(), CcmConvergenceError> {
        self.arithmetic_precision.validate("arithmetic-precision")?;
        self.mode_resolution.validate("mode-resolution")?;
        self.prime_or_continuum_accuracy
            .validate("prime-or-continuum")?;
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModeStarvationPoint {
    pub n_modes: usize,
    pub prime_cutoff: String,
    pub d_n_modes: f64,
    pub raw_mode_error_digits: f64,
    pub prime_ceiling_digits: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModeStarvationReport {
    pub points: Vec<ModeStarvationPoint>,
    pub first_starved_index: Option<usize>,
    pub starvation_detected: bool,
}

pub fn analyze_mode_starvation(
    points: Vec<ModeStarvationPoint>,
) -> Result<ModeStarvationReport, CcmConvergenceError> {
    if points.len() < 2 {
        return Err(CcmConvergenceError(
            "mode-starvation analysis requires at least two mode budgets".to_owned(),
        ));
    }
    for (index, point) in points.iter().enumerate() {
        if point.n_modes == 0
            || point.prime_cutoff.trim().is_empty()
            || !point.d_n_modes.is_finite()
            || !point.raw_mode_error_digits.is_finite()
            || !point.prime_ceiling_digits.is_finite()
            || point.d_n_modes < 0.0
            || point.raw_mode_error_digits < 0.0
            || point.prime_ceiling_digits < 0.0
            || (index > 0 && point.n_modes <= points[index - 1].n_modes)
        {
            return Err(CcmConvergenceError(
                "mode-starvation points must have increasing modes and finite digit budgets"
                    .to_owned(),
            ));
        }
    }
    let first_starved_index = points.iter().position(|point| {
        point.d_n_modes.min(point.raw_mode_error_digits) < point.prime_ceiling_digits
    });
    Ok(ModeStarvationReport {
        starvation_detected: first_starved_index.is_some(),
        first_starved_index,
        points,
    })
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PrimeFloorPoint {
    pub cutoff: String,
    pub cutoff_is_prime: bool,
    pub prime_power_count: usize,
    pub largest_prime_power: u64,
    pub smooth_leading_digits: f64,
    pub measured_digits: f64,
    pub arithmetic_modulation_digits: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PrimeFloorReport {
    pub points: Vec<PrimeFloorPoint>,
    pub contains_nonprime_cutoff: bool,
    pub maximum_absolute_modulation_digits: f64,
}

pub fn analyze_prime_floor(
    mut points: Vec<PrimeFloorPoint>,
) -> Result<PrimeFloorReport, CcmConvergenceError> {
    if points.len() < 2 {
        return Err(CcmConvergenceError(
            "prime-floor analysis requires at least two cutoffs".to_owned(),
        ));
    }
    for point in &mut points {
        if point.cutoff.trim().is_empty()
            || point.prime_power_count == 0
            || point.largest_prime_power < 2
            || !point.smooth_leading_digits.is_finite()
            || !point.measured_digits.is_finite()
        {
            return Err(CcmConvergenceError(
                "prime-floor points require cutoff and finite prime-power diagnostics".to_owned(),
            ));
        }
        point.arithmetic_modulation_digits = point.measured_digits - point.smooth_leading_digits;
    }
    let contains_nonprime_cutoff = points.iter().any(|point| !point.cutoff_is_prime);
    if !contains_nonprime_cutoff {
        return Err(CcmConvergenceError(
            "prime-floor experiment must include a nonprime cutoff".to_owned(),
        ));
    }
    let maximum_absolute_modulation_digits = points
        .iter()
        .map(|point| point.arithmetic_modulation_digits.abs())
        .fold(0.0_f64, f64::max);
    Ok(PrimeFloorReport {
        points,
        contains_nonprime_cutoff,
        maximum_absolute_modulation_digits,
    })
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PrecisionSweepPoint {
    pub precision_bits: u32,
    pub measured_digits: f64,
    pub agreement_with_previous_digits: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PrecisionSweepReport {
    pub requested_digits: u32,
    pub points: Vec<PrecisionSweepPoint>,
    pub slope_digits_per_bit: Vec<f64>,
    pub requested_digits_confirmed: bool,
}

pub fn analyze_precision_sweep(
    requested_digits: u32,
    points: Vec<PrecisionSweepPoint>,
) -> Result<PrecisionSweepReport, CcmConvergenceError> {
    if requested_digits == 0 || points.len() < 2 {
        return Err(CcmConvergenceError(
            "precision sweep requires requested digits and a higher-precision repeat".to_owned(),
        ));
    }
    for (index, point) in points.iter().enumerate() {
        if point.precision_bits < 53
            || !point.measured_digits.is_finite()
            || point.measured_digits < 0.0
            || (index > 0 && point.precision_bits <= points[index - 1].precision_bits)
            || point
                .agreement_with_previous_digits
                .is_some_and(|digits| !digits.is_finite() || digits < 0.0)
        {
            return Err(CcmConvergenceError(
                "precision sweep points must increase in precision with finite diagnostics"
                    .to_owned(),
            ));
        }
    }
    let slope_digits_per_bit = points
        .windows(2)
        .map(|pair| {
            (pair[1].measured_digits - pair[0].measured_digits)
                / f64::from(pair[1].precision_bits - pair[0].precision_bits)
        })
        .collect::<Vec<_>>();
    let requested = f64::from(requested_digits);
    let requested_digits_confirmed = points.last().is_some_and(|point| {
        point.measured_digits >= requested
            && point
                .agreement_with_previous_digits
                .is_some_and(|agreement| agreement >= requested)
    });
    Ok(PrecisionSweepReport {
        requested_digits,
        points,
        slope_digits_per_bit,
        requested_digits_confirmed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_budget_report_and_experiment_diagnostics_round_trip() {
        let budgets = CcmConvergenceBudgetReport {
            arithmetic_precision: ConvergenceBudgetAssessment {
                estimated_digits: 70.0,
                measured_digits: 65.0,
                status: BudgetStatus::Satisfied,
                diagnostic: "256-bit run repeated at 384 bits".to_owned(),
            },
            mode_resolution: ConvergenceBudgetAssessment {
                estimated_digits: 25.0,
                measured_digits: 22.0,
                status: BudgetStatus::Unsatisfied,
                diagnostic: "raw mode error is the active ceiling".to_owned(),
            },
            prime_or_continuum_accuracy: ConvergenceBudgetAssessment {
                estimated_digits: 40.0,
                measured_digits: 37.0,
                status: BudgetStatus::Satisfied,
                diagnostic: "prime-power floor measured at fixed mode budget".to_owned(),
            },
        };
        budgets.validate().unwrap();

        let starvation = analyze_mode_starvation(vec![
            ModeStarvationPoint {
                n_modes: 20,
                prime_cutoff: "100".to_owned(),
                d_n_modes: 12.0,
                raw_mode_error_digits: 10.0,
                prime_ceiling_digits: 30.0,
            },
            ModeStarvationPoint {
                n_modes: 40,
                prime_cutoff: "100".to_owned(),
                d_n_modes: 28.0,
                raw_mode_error_digits: 26.0,
                prime_ceiling_digits: 30.0,
            },
        ])
        .unwrap();
        assert!(starvation.starvation_detected);

        let prime = analyze_prime_floor(vec![
            PrimeFloorPoint {
                cutoff: "12.5".to_owned(),
                cutoff_is_prime: false,
                prime_power_count: 8,
                largest_prime_power: 11,
                smooth_leading_digits: 8.0,
                measured_digits: 7.25,
                arithmetic_modulation_digits: 99.0,
            },
            PrimeFloorPoint {
                cutoff: "13".to_owned(),
                cutoff_is_prime: true,
                prime_power_count: 9,
                largest_prime_power: 13,
                smooth_leading_digits: 8.2,
                measured_digits: 8.6,
                arithmetic_modulation_digits: 99.0,
            },
        ])
        .unwrap();
        assert!(prime.contains_nonprime_cutoff);
        assert_eq!(prime.points[0].arithmetic_modulation_digits, -0.75);

        let precision = analyze_precision_sweep(
            50,
            vec![
                PrecisionSweepPoint {
                    precision_bits: 192,
                    measured_digits: 48.0,
                    agreement_with_previous_digits: None,
                },
                PrecisionSweepPoint {
                    precision_bits: 320,
                    measured_digits: 80.0,
                    agreement_with_previous_digits: Some(60.0),
                },
            ],
        )
        .unwrap();
        assert!(precision.requested_digits_confirmed);
        assert_eq!(precision.slope_digits_per_bit, vec![0.25]);

        let encoded = serde_json::to_string(&(budgets, starvation, prime, precision)).unwrap();
        let _: (
            CcmConvergenceBudgetReport,
            ModeStarvationReport,
            PrimeFloorReport,
            PrecisionSweepReport,
        ) = serde_json::from_str(&encoded).unwrap();
    }

    #[test]
    fn incomplete_experiments_fail_closed() {
        assert!(analyze_mode_starvation(Vec::new()).is_err());
        assert!(analyze_prime_floor(vec![PrimeFloorPoint {
            cutoff: "13".to_owned(),
            cutoff_is_prime: true,
            prime_power_count: 1,
            largest_prime_power: 13,
            smooth_leading_digits: 1.0,
            measured_digits: 1.0,
            arithmetic_modulation_digits: 0.0,
        }])
        .is_err());
        assert!(analyze_precision_sweep(50, Vec::new()).is_err());
    }
}
