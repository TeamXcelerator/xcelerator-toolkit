use crate::{
    AssuranceLevel, ConfigError, DecimalLiteral, EigenTarget, PrecisionPolicy, Reproducibility,
    Subspace,
};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// Backend-neutral stopping thresholds.
///
/// Thresholds are serialized as finite decimal literals so a request may ask
/// for residuals far below the f64 range. A concrete solver parses them into
/// its active scalar backend; an explicitly f64-only solver may reject a value
/// outside the finite f64 range.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoppingPolicy {
    pub absolute_residual: DecimalLiteral,
    pub scaled_backward_error: DecimalLiteral,
    pub maximum_iterations: usize,
    pub minimum_iterations: usize,
}

impl Default for StoppingPolicy {
    fn default() -> Self {
        Self {
            absolute_residual: DecimalLiteral::new("1e-30").expect("default residual is valid"),
            scaled_backward_error: DecimalLiteral::new("1e-30")
                .expect("default backward error is valid"),
            maximum_iterations: 500,
            minimum_iterations: 2,
        }
    }
}

impl StoppingPolicy {
    pub fn validate(&self) -> Result<(), ConfigError> {
        let zero = DecimalLiteral::new("0").expect("zero is valid");
        for (name, value) in [
            ("absolute_residual", &self.absolute_residual),
            ("scaled_backward_error", &self.scaled_backward_error),
        ] {
            value.validate()?;
            if value.cmp_numeric(&zero)? != Ordering::Greater {
                return Err(ConfigError::new(format!(
                    "{name} must be strictly positive"
                )));
            }
        }
        if self.maximum_iterations == 0 {
            return Err(ConfigError::new("maximum_iterations must be positive"));
        }
        if self.minimum_iterations > self.maximum_iterations {
            return Err(ConfigError::new(
                "minimum_iterations may not exceed maximum_iterations",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolverConfig {
    pub target: EigenTarget,
    pub subspace: Subspace,
    pub assurance: AssuranceLevel,
    pub precision: PrecisionPolicy,
    pub stopping: StoppingPolicy,
    pub reproducibility: Reproducibility,
    pub algorithm_preferences: Vec<String>,
    pub allow_lower_precision_seed: bool,
    pub allow_randomized_seed: bool,
}

/// Provenance class for data supplied only to initialize a solver.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SeedGeneratorKind {
    Deterministic,
    CachedHp,
    LowerPrecision,
    Randomized,
}

/// Evidence that a generated seed remained non-decisive at acceptance.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeedUseEvidence {
    pub generator: SeedGeneratorKind,
    pub used_only_as_initial_guess: bool,
    pub final_hp_verification: bool,
    pub acceptance_depended_on_seed: bool,
}

impl SolverConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        self.target.validate()?;
        self.subspace.validate()?;
        self.precision.validate()?;
        self.stopping.validate()?;
        if self.assurance == AssuranceLevel::Certified
            && self.reproducibility == Reproducibility::Exploratory
        {
            return Err(ConfigError::new(
                "certified assurance may not use exploratory reproducibility",
            ));
        }
        Ok(())
    }

    /// Check authorization before a seed generator executes.
    pub fn authorize_seed_generator(
        &self,
        generator: SeedGeneratorKind,
    ) -> Result<(), ConfigError> {
        self.validate()?;
        match generator {
            SeedGeneratorKind::LowerPrecision if !self.allow_lower_precision_seed => Err(
                ConfigError::new("lower-precision seed generation was not explicitly requested"),
            ),
            SeedGeneratorKind::Randomized if !self.allow_randomized_seed => Err(ConfigError::new(
                "randomized seed generation was not explicitly requested",
            )),
            _ => Ok(()),
        }
    }

    /// Validate the completed use of a seed before accepting a solver result.
    pub fn validate_seed_use(&self, evidence: &SeedUseEvidence) -> Result<(), ConfigError> {
        self.authorize_seed_generator(evidence.generator)?;
        if !evidence.used_only_as_initial_guess {
            return Err(ConfigError::new(
                "seed data may be used only as an initial guess",
            ));
        }
        if self.assurance == AssuranceLevel::Certified {
            if !evidence.final_hp_verification {
                return Err(ConfigError::new(
                    "certified acceptance requires final HP verification independent of the seed generator",
                ));
            }
            if evidence.acceptance_depended_on_seed {
                return Err(ConfigError::new(
                    "seed data may not determine a certified acceptance decision",
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stopping_policy_supports_deep_hp_thresholds() {
        let policy = StoppingPolicy {
            absolute_residual: DecimalLiteral::new("1e-10000").unwrap(),
            scaled_backward_error: DecimalLiteral::new("5e-9000").unwrap(),
            maximum_iterations: 100,
            minimum_iterations: 2,
        };
        assert!(policy.validate().is_ok());
    }

    #[test]
    fn stopping_policy_rejects_nonpositive_thresholds() {
        let policy = StoppingPolicy {
            absolute_residual: DecimalLiteral::new("0").unwrap(),
            ..StoppingPolicy::default()
        };
        assert!(policy.validate().is_err());
    }

    fn certified_config() -> SolverConfig {
        SolverConfig {
            target: EigenTarget::AlgebraicLargest,
            subspace: Subspace::Full,
            assurance: AssuranceLevel::Certified,
            precision: PrecisionPolicy::default(),
            stopping: StoppingPolicy::default(),
            reproducibility: Reproducibility::Deterministic,
            algorithm_preferences: Vec::new(),
            allow_lower_precision_seed: false,
            allow_randomized_seed: false,
        }
    }

    #[test]
    fn seed_generators_require_explicit_authorization() {
        let config = certified_config();
        assert!(config
            .authorize_seed_generator(SeedGeneratorKind::Randomized)
            .is_err());
        assert!(config
            .authorize_seed_generator(SeedGeneratorKind::LowerPrecision)
            .is_err());
        assert!(config
            .authorize_seed_generator(SeedGeneratorKind::Deterministic)
            .is_ok());
    }

    #[test]
    fn certified_seed_use_requires_independent_final_hp_acceptance() {
        let mut config = certified_config();
        config.allow_randomized_seed = true;
        let valid = SeedUseEvidence {
            generator: SeedGeneratorKind::Randomized,
            used_only_as_initial_guess: true,
            final_hp_verification: true,
            acceptance_depended_on_seed: false,
        };
        assert!(config.validate_seed_use(&valid).is_ok());

        let mut decisive = valid.clone();
        decisive.acceptance_depended_on_seed = true;
        assert!(config.validate_seed_use(&decisive).is_err());
        let mut unverified = valid;
        unverified.final_hp_verification = false;
        assert!(config.validate_seed_use(&unverified).is_err());
    }

    #[test]
    fn certified_acceptance_remains_valid_with_randomized_seeding_disabled() {
        let config = certified_config();
        let cold = SeedUseEvidence {
            generator: SeedGeneratorKind::Deterministic,
            used_only_as_initial_guess: true,
            final_hp_verification: true,
            acceptance_depended_on_seed: false,
        };
        assert!(config.validate_seed_use(&cold).is_ok());
    }
}
