// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Versioned, shared capture recipes. Capture volume, numerical algorithms,
//! source acquisition, and certification are separate policies. Persist this
//! resolved plan for historical replay rather than re-resolving a level name.
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

pub const CAPTURE_PLAN_SEMANTICS: &str = "ccm-measurement-capture-plan-v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CcmCaptureLevel {
    Claim,
    Research,
    Gap,
    Maximum,
    Ultra,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CcmCapturePlan {
    pub schema_version: u32,
    pub semantics: String,
    pub level: CcmCaptureLevel,
    pub source_even_dimension: usize,
    pub sector_eigenpairs: Option<usize>,
    pub capture_prime_power_response: bool,
    pub capture_u_flow_response: bool,
    pub capture_prefix_analysis: bool,
    pub prefix_checkpoint_dimensions: Vec<usize>,
    pub requires_even_sector: bool,
    pub certification_requested: bool,
    pub changes_numerical_algorithm: bool,
    pub missing_diagnostic_sources: String,
}
impl CcmCapturePlan {
    pub fn resolve(
        level: CcmCaptureLevel,
        maximum_eigenpairs: usize,
        source_even_dimension: usize,
    ) -> Result<Self> {
        if source_even_dimension == 0 || source_even_dimension > 8193 || maximum_eigenpairs < 2 {
            bail!("invalid capture dimensions or sector eigenpair count");
        }
        let ultra = level == CcmCaptureLevel::Ultra;
        Ok(Self {
            schema_version: 1,
            semantics: CAPTURE_PLAN_SEMANTICS.into(),
            level,
            source_even_dimension,
            sector_eigenpairs: match level {
                CcmCaptureLevel::Claim | CcmCaptureLevel::Research => None,
                CcmCaptureLevel::Gap => Some(2),
                _ => Some(maximum_eigenpairs),
            },
            capture_prime_power_response: ultra,
            capture_u_flow_response: ultra,
            capture_prefix_analysis: ultra,
            prefix_checkpoint_dimensions: if ultra {
                vec![source_even_dimension]
            } else {
                vec![]
            },
            requires_even_sector: ultra,
            certification_requested: false,
            changes_numerical_algorithm: false,
            missing_diagnostic_sources: "report_missing_never_compute_a_replacement_source".into(),
        })
    }
    /// Top-tier measurements. This does not request either interval certificate
    /// route, alternative prime arithmetic, or a new quadrature-order policy.
    pub fn ultra(maximum_eigenpairs: usize, source_even_dimension: usize) -> Result<Self> {
        Self::resolve(
            CcmCaptureLevel::Ultra,
            maximum_eigenpairs,
            source_even_dimension,
        )
    }
    pub fn validate(&self) -> Result<()> {
        let ultra = self.level == CcmCaptureLevel::Ultra;
        let sector_valid = match self.level {
            CcmCaptureLevel::Claim | CcmCaptureLevel::Research => self.sector_eigenpairs.is_none(),
            CcmCaptureLevel::Gap => self.sector_eigenpairs == Some(2),
            CcmCaptureLevel::Maximum | CcmCaptureLevel::Ultra => {
                self.sector_eigenpairs.is_some_and(|n| n >= 2)
            }
        };
        if !sector_valid
            || self.capture_prime_power_response != ultra
            || self.capture_u_flow_response != ultra
            || (ultra && !self.capture_prefix_analysis)
            || self.requires_even_sector != (ultra || self.capture_prefix_analysis)
            || self.schema_version != 1
            || self.semantics != CAPTURE_PLAN_SEMANTICS
            || self.certification_requested
            || self.changes_numerical_algorithm
            || self.source_even_dimension == 0
            || self.source_even_dimension > 8193
            || self
                .prefix_checkpoint_dimensions
                .windows(2)
                .any(|p| p[0] >= p[1])
            || self
                .prefix_checkpoint_dimensions
                .iter()
                .any(|&k| k == 0 || k > self.source_even_dimension)
            || (!self.capture_prefix_analysis && !self.prefix_checkpoint_dimensions.is_empty())
            || self.missing_diagnostic_sources
                != "report_missing_never_compute_a_replacement_source"
        {
            bail!("unsupported or inconsistent measurement capture plan");
        }
        Ok(())
    }
    /// Make an explicit prefix request at a lower capture level, or select
    /// campaign checkpoints for ultra. Missing eigenstates remain missing.
    pub fn with_prefix_checkpoints(mut self, checkpoints: Vec<usize>) -> Result<Self> {
        self.capture_prefix_analysis = true;
        self.requires_even_sector = true;
        self.prefix_checkpoint_dimensions = checkpoints;
        self.validate()?;
        Ok(self)
    }

    #[cfg(feature = "hp")]
    pub fn primary_options(&self) -> Result<super::hp::CcmResearchCaptureOptions> {
        use super::hp::{CcmResearchCaptureOptions, CcmSectorAnalysisOptions};
        self.validate()?;
        let mut options = match self.level {
            CcmCaptureLevel::Maximum | CcmCaptureLevel::Ultra => {
                CcmResearchCaptureOptions::maximum(self.sector_eigenpairs.unwrap_or(2))
            }
            _ => CcmResearchCaptureOptions {
                capture_evenness: self.level == CcmCaptureLevel::Gap,
                sector_analysis: if self.level == CcmCaptureLevel::Gap {
                    Some(CcmSectorAnalysisOptions::selected(2))
                } else {
                    None
                },
                sector_gap_certification: None,
                root_certification: None,
                distance_capture: None,
                capture_prime_power_response: false,
                capture_u_flow_response: false,
            },
        };
        if self.level == CcmCaptureLevel::Ultra {
            if let Some(distance) = options.distance_capture.take() {
                options.distance_capture = Some(distance.with_deviation_decomposition());
            }
        }
        options.capture_prime_power_response = self.capture_prime_power_response;
        options.capture_u_flow_response = self.capture_u_flow_response;
        Ok(options)
    }

    /// Execute the retained-source phase of this capture recipe. The
    /// application first runs primary_options through its established solver
    /// path, then supplies those exact approved retained sources here.
    #[cfg(feature = "hp")]
    pub fn capture_retained_diagnostics(
        &self,
        matrix: &super::prefix::RetainedEvenMatrix,
        eigenpairs: &[super::prefix::RetainedEvenEigenpair],
        cache: &xc_cache::ArtifactCacheContext<'_>,
    ) -> Result<Option<xc_cache::ArtifactExecutionCacheResult<super::prefix::CcmPrefixAnalysis>>>
    {
        if matrix.dimension() != self.source_even_dimension {
            bail!("retained matrix does not match the resolved capture dimension");
        }
        match self.prefix_options(matrix.source_precision_bits())? {
            Some(options) => super::prefix::analyze_retained_prefixes_via_cache(
                matrix, &options, eigenpairs, cache,
            )
            .map(Some),
            None => Ok(None),
        }
    }

    /// Companion options for the retained-source diagnostic phase. Calling
    /// primary_options alone does NOT execute this phase. This two-phase API
    /// preserves source reuse and keeps old capture structs backward-compatible.
    #[cfg(feature = "hp")]
    pub fn prefix_options(
        &self,
        source_precision_bits: u32,
    ) -> Result<Option<super::prefix::PrefixAnalysisOptions>> {
        self.validate()?;
        if !self.capture_prefix_analysis {
            return Ok(None);
        }
        if !(64..=1_000_000).contains(&source_precision_bits) {
            bail!("unsupported prefix source precision");
        }
        let full = xc_numerics::reduction::roundtrip_decimal_digits(source_precision_bits);
        let mut widths = vec![80, 96, 112, full];
        widths.sort_unstable();
        widths.dedup();
        widths.retain(|d| *d <= full);
        if widths.is_empty() {
            widths.push(full);
        }
        // A conservative export screen chosen from arithmetic precision,
        // explicitly not an estimate of matrix-assembly accuracy.
        let digits = ((source_precision_bits.saturating_sub(48) as usize) * 3 / 10).max(1);
        Ok(Some(super::prefix::PrefixAnalysisOptions {
            working_precision_bits: source_precision_bits,
            pivot_margin_bits: 32,
            checkpoint_dimensions: self.prefix_checkpoint_dimensions.clone(),
            export_significant_digits: widths,
            export_relative_tolerance: format!("1e-{digits}"),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn only_ultra_adds_prefixes_by_default_and_never_changes_algorithms() {
        for level in [
            CcmCaptureLevel::Claim,
            CcmCaptureLevel::Research,
            CcmCaptureLevel::Gap,
            CcmCaptureLevel::Maximum,
            CcmCaptureLevel::Ultra,
        ] {
            let plan = CcmCapturePlan::resolve(level, 8, 33).unwrap();
            plan.validate().unwrap();
            assert_eq!(
                plan.capture_prefix_analysis,
                level == CcmCaptureLevel::Ultra
            );
            assert!(!plan.certification_requested);
            assert!(!plan.changes_numerical_algorithm);
            assert_eq!(
                plan,
                serde_json::from_slice(&serde_json::to_vec(&plan).unwrap()).unwrap()
            );
        }
    }
    #[test]
    fn bad_plans_and_out_of_range_checkpoints_are_rejected() {
        assert!(CcmCapturePlan::ultra(8, 0).is_err());
        assert!(CcmCapturePlan::ultra(8, 4)
            .unwrap()
            .with_prefix_checkpoints(vec![2, 2])
            .is_err());
        let mut plan = CcmCapturePlan::ultra(8, 4).unwrap();
        plan.changes_numerical_algorithm = true;
        assert!(plan.validate().is_err());
    }
    #[cfg(feature = "hp")]
    #[test]
    fn ultra_has_two_explicit_phases_without_certification() {
        let plan = CcmCapturePlan::ultra(8, 33).unwrap();
        let p = plan.primary_options().unwrap();
        assert!(p.capture_prime_power_response && p.capture_u_flow_response);
        assert!(p.root_certification.is_none() && p.sector_gap_certification.is_none());
        assert!(p.distance_capture.unwrap().capture_deviation_decomposition);
        let prefix = plan.prefix_options(256).unwrap().unwrap();
        assert_eq!(prefix.checkpoint_dimensions, vec![33]);
    }
}
