// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Saved-configuration reproduction for complete finite CCM observations.

use super::{run_f64, CcmParams};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use xc_cache::ContentDigest;
use xc_core::{ConfigDigest, ExecutionFingerprint, SolverProvenance};

const SOLVER_ID: &str = "ccm_f64_tau_symmetric_eigen_v1";

/// Complete mathematical configuration for the reproducible f64 CCM route.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CcmF64ObservationConfig {
    pub schema_version: u32,
    pub lambda_squared_integer: u64,
    pub n_modes: usize,
    pub finite_scope_statement: String,
}

impl CcmF64ObservationConfig {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != 1
            || self.lambda_squared_integer <= 1
            || self.n_modes == 0
            || self.finite_scope_statement.trim().is_empty()
        {
            bail!("invalid reproducible f64 CCM observation configuration");
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<ConfigDigest> {
        self.validate()?;
        let bytes = serde_json::to_vec(self).context("serialize CCM observation configuration")?;
        Ok(ConfigDigest(ContentDigest::sha256(&bytes).0))
    }
}

/// Timing-independent numerical payload of one complete finite CCM run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CcmF64NumericalObservation {
    pub eigenvalues_positive: Vec<f64>,
    pub weil_minimum_eigenvalue: f64,
    pub normalized_even_source: Vec<f64>,
}

impl CcmF64NumericalObservation {
    fn validate_for(&self, config: &CcmF64ObservationConfig) -> Result<()> {
        if self.normalized_even_source.len() != 2 * config.n_modes + 1
            || self
                .normalized_even_source
                .iter()
                .chain(&self.eigenvalues_positive)
                .any(|value| !value.is_finite())
            || !self.weil_minimum_eigenvalue.is_finite()
            || self
                .eigenvalues_positive
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            bail!("reproduced CCM numerical payload is incomplete or noncanonical");
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<ContentDigest> {
        let bytes = serde_json::to_vec(self).context("serialize CCM numerical observation")?;
        Ok(ContentDigest::sha256(&bytes))
    }
}

/// Self-contained saved record used as the input to a later reproduction.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SavedCcmF64Observation {
    pub schema_version: u32,
    pub configuration: CcmF64ObservationConfig,
    pub provenance: SolverProvenance,
    pub numerical_observation: CcmF64NumericalObservation,
    pub numerical_digest: ContentDigest,
    pub elapsed_seconds_observed: f64,
}

impl SavedCcmF64Observation {
    pub fn validate(&self) -> Result<()> {
        self.configuration.validate()?;
        self.provenance
            .validate_saved_result()
            .context("validate saved CCM provenance")?;
        validate_provenance(&self.configuration, &self.provenance)?;
        self.numerical_observation
            .validate_for(&self.configuration)?;
        if self.schema_version != 1
            || !self.numerical_digest.validate()
            || self.numerical_observation.digest()? != self.numerical_digest
            || !self.elapsed_seconds_observed.is_finite()
            || self.elapsed_seconds_observed < 0.0
        {
            bail!("saved CCM observation identity or telemetry is invalid");
        }
        Ok(())
    }
}

/// Evidence emitted after recomputing a saved record from configuration alone.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CcmF64ReproductionReport {
    pub schema_version: u32,
    pub original_numerical_digest: ContentDigest,
    pub reproduced_numerical_digest: ContentDigest,
    pub execution_fingerprints_match: bool,
    pub numerically_identical: bool,
    pub reproduced_elapsed_seconds: f64,
    pub finite_scope_statement: String,
}

fn solver_configuration() -> serde_json::Value {
    serde_json::json!({
        "algorithm": SOLVER_ID,
        "precision_bits": 53,
        "scalar_backend": "f64",
        "spectrum_root_method": "pole_interval_bisection"
    })
}

fn validate_provenance(
    config: &CcmF64ObservationConfig,
    provenance: &SolverProvenance,
) -> Result<()> {
    let expected_domain = serde_json::to_value(config).context("serialize CCM domain config")?;
    let fingerprint = provenance
        .execution_fingerprint
        .as_ref()
        .context("saved CCM provenance lacks an execution fingerprint")?;
    if provenance.scalar_backend != "f64"
        || provenance.precision_bits != Some(53)
        || provenance.solver_configuration.as_ref() != Some(&solver_configuration())
        || provenance.domain_configuration.as_ref() != Some(&expected_domain)
        || fingerprint.effective_configuration_digest != config.digest()?
    {
        bail!("saved CCM provenance does not bind the executable configuration");
    }
    Ok(())
}

fn execute(config: &CcmF64ObservationConfig) -> Result<(CcmF64NumericalObservation, f64)> {
    config.validate()?;
    let params = CcmParams::from_lambda_sq_integer(config.lambda_squared_integer, config.n_modes);
    let result = run_f64(&params).context("execute saved f64 CCM observation")?;
    let numerical = CcmF64NumericalObservation {
        eigenvalues_positive: result.eigenvalues_pos,
        weil_minimum_eigenvalue: result.weil_min_eigenvalue,
        normalized_even_source: result.xi,
    };
    numerical.validate_for(config)?;
    Ok((numerical, result.elapsed_seconds))
}

/// Executes and saves a complete finite observation with bound provenance.
pub fn run_saved_ccm_f64_observation(
    configuration: CcmF64ObservationConfig,
    provenance: SolverProvenance,
) -> Result<SavedCcmF64Observation> {
    provenance
        .validate_saved_result()
        .context("validate CCM saved-result provenance")?;
    validate_provenance(&configuration, &provenance)?;
    let (numerical_observation, elapsed_seconds_observed) = execute(&configuration)?;
    let numerical_digest = numerical_observation.digest()?;
    let record = SavedCcmF64Observation {
        schema_version: 1,
        configuration,
        provenance,
        numerical_observation,
        numerical_digest,
        elapsed_seconds_observed,
    };
    record.validate()?;
    Ok(record)
}

/// Recomputes a saved finite CCM observation from its bound configuration.
///
/// # Mathematical semantics
/// Rebuilds the Weil matrix, selects its smallest finite eigenpair, normalizes
/// the even source, and rediscovers the finite rational spectrum. The stored
/// numerical values are used only after execution for comparison.
///
/// # Precision
/// This route is explicitly IEEE-754 binary64. Exact replay is allowed only
/// when the current and saved bitwise execution fingerprints match.
///
/// # Failure states
/// Invalid or tampered configuration, provenance, digest, execution environment,
/// numerical construction, or replay mismatch returns an error and no success
/// report.
///
/// # Assurance and validity
/// A successful report proves exact reproduction of this finite observation in
/// the recorded environment. It does not certify an HP result, a limiting
/// operator, RH, or any other infinite-dimensional claim.
///
/// # Cache effects
/// Replay computes locally without cache reads, writes, or publication. Saved
/// cache inputs, if any, remain independently bound in provenance.
///
/// # Example
/// Compiled example: `crates/xc-spectral/examples/ccm_reproduce.rs`.
pub fn reproduce_saved_ccm_f64_observation(
    saved: &SavedCcmF64Observation,
    current_execution: &ExecutionFingerprint,
) -> Result<CcmF64ReproductionReport> {
    saved.validate()?;
    let saved_execution = saved
        .provenance
        .execution_fingerprint
        .as_ref()
        .context("saved CCM provenance lacks its execution fingerprint")?;
    let comparison = saved_execution
        .comparison_plan(current_execution)
        .context("compare saved and current CCM execution fingerprints")?;
    if !comparison.byte_identity_permitted {
        bail!("current execution fingerprint differs from the saved bitwise CCM environment");
    }
    let (reproduced, reproduced_elapsed_seconds) = execute(&saved.configuration)?;
    let reproduced_numerical_digest = reproduced.digest()?;
    if reproduced != saved.numerical_observation
        || reproduced_numerical_digest != saved.numerical_digest
    {
        bail!("reproduced CCM observation differs from the saved numerical payload");
    }
    Ok(CcmF64ReproductionReport {
        schema_version: 1,
        original_numerical_digest: saved.numerical_digest.clone(),
        reproduced_numerical_digest,
        execution_fingerprints_match: true,
        numerically_identical: true,
        reproduced_elapsed_seconds,
        finite_scope_statement: saved.configuration.finite_scope_statement.clone(),
    })
}

/// Canonical solver configuration to bind into `SolverProvenance`.
pub fn ccm_f64_reproduction_solver_configuration() -> serde_json::Value {
    solver_configuration()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};
    use xc_core::{
        ExecutionFingerprint, PrecisionFingerprint, Reproducibility, ThreadPolicyFingerprint,
    };

    fn configuration() -> CcmF64ObservationConfig {
        CcmF64ObservationConfig {
            schema_version: 1,
            lambda_squared_integer: 5,
            n_modes: 2,
            finite_scope_statement:
                "finite f64 CCM observation; not evidence of an infinite-dimensional theorem"
                    .to_owned(),
        }
    }

    fn provenance(config: &CcmF64ObservationConfig) -> SolverProvenance {
        let fingerprint = ExecutionFingerprint {
            schema_version: 1,
            toolkit_revision: "reproduction-test-revision".to_owned(),
            dependency_revisions: BTreeMap::from([(
                "nalgebra".to_owned(),
                "Cargo.lock".to_owned(),
            )]),
            compiler: "rustc-test".to_owned(),
            target_triple: "portable-test-target".to_owned(),
            native_libraries: BTreeMap::new(),
            scalar_backend: "f64".to_owned(),
            scalar_backend_version: "ieee-754-binary64".to_owned(),
            precision: PrecisionFingerprint {
                working_precision_bits: 53,
                guard_bits: 0,
                rounding_policy: "round-to-nearest-ties-even".to_owned(),
            },
            algorithm_semantics_versions: BTreeMap::from([(
                "ccm_observation".to_owned(),
                SOLVER_ID.to_owned(),
            )]),
            cpu_feature_policy: "portable".to_owned(),
            thread_policy: ThreadPolicyFingerprint {
                thread_count: 1,
                scheduling_policy: "sequential".to_owned(),
                reduction_policy: "fixed-index-order".to_owned(),
            },
            feature_flags: BTreeSet::new(),
            effective_configuration_digest: config.digest().unwrap(),
            resolved_resource_policy_digest: ConfigDigest("a".repeat(64)),
            reproducibility: Reproducibility::Bitwise,
        };
        SolverProvenance::current_package("f64")
            .with_saved_result_context(
                &fingerprint,
                "b".repeat(64),
                "portable-test-platform",
                solver_configuration(),
                serde_json::to_value(config).unwrap(),
            )
            .unwrap()
    }

    #[test]
    fn saved_configuration_and_provenance_reproduce_complete_ccm_observation() {
        let config = configuration();
        let saved = run_saved_ccm_f64_observation(config.clone(), provenance(&config)).unwrap();
        let bytes = serde_json::to_vec(&saved).unwrap();
        let decoded: SavedCcmF64Observation = serde_json::from_slice(&bytes).unwrap();
        decoded.validate().unwrap();
        let current = decoded.provenance.execution_fingerprint.clone().unwrap();
        let report = reproduce_saved_ccm_f64_observation(&decoded, &current).unwrap();
        assert!(report.numerically_identical);
        assert!(report.execution_fingerprints_match);
        assert_eq!(report.original_numerical_digest, saved.numerical_digest);
        assert_eq!(
            report.original_numerical_digest,
            report.reproduced_numerical_digest
        );
    }

    #[test]
    fn reproduction_rejects_tampered_configuration_or_saved_answer() {
        let config = configuration();
        let saved = run_saved_ccm_f64_observation(config.clone(), provenance(&config)).unwrap();

        let mut changed_config = saved.clone();
        changed_config.configuration.n_modes += 1;
        assert!(changed_config.validate().is_err());

        let mut changed_answer = saved;
        changed_answer.numerical_observation.weil_minimum_eigenvalue += 1.0;
        assert!(changed_answer.validate().is_err());
    }

    #[test]
    fn reproduction_rejects_a_different_execution_environment() {
        let config = configuration();
        let saved = run_saved_ccm_f64_observation(config.clone(), provenance(&config)).unwrap();
        let mut current = saved.provenance.execution_fingerprint.clone().unwrap();
        current.compiler = "different-rustc".to_owned();
        assert!(reproduce_saved_ccm_f64_observation(&saved, &current).is_err());
    }
}
