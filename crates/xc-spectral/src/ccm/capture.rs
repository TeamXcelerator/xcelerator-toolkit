// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Versioned, shared capture recipes. Capture volume, numerical algorithms,
//! source acquisition, and certification are separate policies. Persist this
//! resolved plan for historical replay rather than re-resolving a level name.
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
pub use xc_core::PrefixDiagnosticPolicy;

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
    /// Optional arithmetic precision for retained diagnostics. Source bytes and
    /// their assembly precision remain unchanged and independently recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix_working_precision_bits: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix_export_policy: Option<PrefixExportPolicy>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix_diagnostics: Option<PrefixDiagnosticPolicy>,
    pub requires_even_sector: bool,
    pub certification_requested: bool,
    pub changes_numerical_algorithm: bool,
    pub missing_diagnostic_sources: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrefixExportPolicy {
    pub pivot_margin_bits: u32,
    pub significant_digits: Vec<usize>,
    pub relative_tolerance: String,
}
/// Explicit budget for the cubic retained-reduction diagnostic.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RetainedReductionRequest {
    pub working_precision_bits: u32,
    pub maximum_dimension: usize,
    pub relative_tolerance: String,
}

impl CcmCapturePlan {
    fn resolved_value(&self) -> Result<serde_json::Value> {
        let mut canonical = self.clone();
        if let Some(policy) = &mut canonical.prefix_export_policy {
            policy.relative_tolerance = xc_core::DecimalLiteral::new(&policy.relative_tolerance)?
                .canonical()?
                .to_string();
        }
        if canonical
            .prefix_diagnostics
            .is_some_and(|p| p.is_legacy_default())
        {
            canonical.prefix_diagnostics = None;
        }
        Ok(serde_json::to_value(canonical)?)
    }
    pub fn resolve(
        level: CcmCaptureLevel,
        maximum_eigenpairs: usize,
        source_even_dimension: usize,
    ) -> Result<Self> {
        if source_even_dimension == 0
            || source_even_dimension > 8193
            || (!matches!(level, CcmCaptureLevel::Claim | CcmCaptureLevel::Research)
                && maximum_eigenpairs < 2)
        {
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
            prefix_working_precision_bits: None,
            prefix_export_policy: None,
            prefix_diagnostics: ultra.then(PrefixDiagnosticPolicy::full),
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
        if let Some(policy) = &self.prefix_export_policy {
            if !self.capture_prefix_analysis
                || policy.pivot_margin_bits
                    >= self.prefix_working_precision_bits.unwrap_or(1_000_000)
                || policy.significant_digits.is_empty()
                || policy
                    .significant_digits
                    .iter()
                    .any(|&d| d == 0 || d > 1_000_000)
                || policy.significant_digits.windows(2).any(|w| w[0] >= w[1])
            {
                bail!("invalid prefix export policy");
            }
            let tolerance = xc_core::DecimalLiteral::new(&policy.relative_tolerance)?;
            if tolerance.cmp_numeric(&xc_core::DecimalLiteral::new("0")?)?
                != std::cmp::Ordering::Greater
                || tolerance.cmp_numeric(&xc_core::DecimalLiteral::new("1")?)?
                    != std::cmp::Ordering::Less
            {
                bail!("prefix export tolerance must lie strictly between zero and one");
            }
        }
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
            || (self.prefix_diagnostics.is_some() && !self.capture_prefix_analysis)
            || self.prefix_working_precision_bits.is_some_and(|bits| {
                !self.capture_prefix_analysis || !(64..=1_000_000).contains(&bits)
            })
            || self.missing_diagnostic_sources
                != "report_missing_never_compute_a_replacement_source"
        {
            bail!("unsupported or inconsistent measurement capture plan");
        }
        Ok(())
    }
    /// Bind checks that require the retained source precision. Without an
    /// override, `validate` cannot resolve a source-dependent pivot margin.
    pub fn validate_for_source_precision(&self, source_precision_bits: u32) -> Result<()> {
        self.validate()?;
        if !self.capture_prefix_analysis {
            return Ok(());
        }
        if !(64..=1_000_000).contains(&source_precision_bits) {
            bail!("unsupported prefix source precision");
        }
        let working = self
            .prefix_working_precision_bits
            .unwrap_or(source_precision_bits);
        if working < source_precision_bits {
            bail!("prefix working precision must be at least source precision");
        }
        if self
            .prefix_export_policy
            .as_ref()
            .is_some_and(|policy| policy.pivot_margin_bits >= working)
        {
            bail!("prefix pivot margin must be below working precision");
        }
        Ok(())
    }
    /// Enumerate the requested diagnostic groups before either capture phase.
    /// A target-dependent group remains pending/missing when its private target
    /// is unavailable. This receipt neither launches work nor requests assurance.
    /// Applications must record each group's actual outcome and persist it.
    pub fn receipt(&self) -> Result<xc_core::CaptureReceipt> {
        self.validate()?;
        let mut requested = Vec::new();
        if matches!(
            self.level,
            CcmCaptureLevel::Gap | CcmCaptureLevel::Maximum | CcmCaptureLevel::Ultra
        ) {
            requested.extend(["evenness", "sector_analysis"].map(str::to_owned));
        }
        if matches!(
            self.level,
            CcmCaptureLevel::Maximum | CcmCaptureLevel::Ultra
        ) {
            requested.extend(
                [
                    "root_conditioning",
                    "distance_profile",
                    "target_distance",
                    "distance_resolution",
                    "target_residual_analysis",
                ]
                .map(str::to_owned),
            );
        }
        if self.level == CcmCaptureLevel::Ultra {
            requested.push("deviation_decomposition".into());
        }
        if self.capture_prime_power_response {
            requested.push("prime_power_response".into());
        }
        if self.capture_u_flow_response {
            requested.push("u_flow_response".into());
        }
        if self.capture_prefix_analysis {
            requested.push("prefix_ladder".into());
            requested.extend(
                self.prefix_checkpoint_dimensions
                    .iter()
                    .map(|k| format!("prefix_checkpoint_{k}")),
            );
        }
        Ok(xc_core::CaptureReceipt::new(
            &self.resolved_value()?,
            requested,
        )?)
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

    /// Increase retained arithmetic precision without acquiring a new source.
    /// The override is part of the persisted plan and its cache identity.
    pub fn with_prefix_working_precision(mut self, bits: u32) -> Result<Self> {
        self.capture_prefix_analysis = true;
        self.requires_even_sector = true;
        self.prefix_working_precision_bits = Some(bits);
        self.validate()?;
        Ok(self)
    }

    pub fn with_prefix_export_policy(mut self, mut policy: PrefixExportPolicy) -> Result<Self> {
        self.capture_prefix_analysis = true;
        self.requires_even_sector = true;
        policy.relative_tolerance = xc_core::DecimalLiteral::new(policy.relative_tolerance)?
            .canonical()?
            .to_string();
        self.prefix_export_policy = Some(policy);
        self.validate()?;
        Ok(self)
    }

    /// Choose additional prefix evidence explicitly. New Ultra plans request
    /// the third moment and cancellation; old serialized plans retain their policy.
    pub fn with_prefix_diagnostics(mut self, policy: PrefixDiagnosticPolicy) -> Result<Self> {
        self.capture_prefix_analysis = true;
        self.requires_even_sector = true;
        self.prefix_diagnostics = (!policy.is_legacy_default()).then_some(policy);
        self.validate()?;
        Ok(self)
    }

    /// Execute every requested primary diagnostic through the supplied adapter,
    /// execute retained prefixes once, and persist all terminal outcomes. The
    /// adapter normally uses the application's existing managed numerical APIs.
    /// Missing retained sources are recorded; they are never regenerated here.
    #[cfg(feature = "hp")]
    pub fn execute_with_receipt<F>(
        &self,
        retained: Option<(
            &super::prefix::RetainedEvenMatrix,
            &[super::prefix::RetainedEvenEigenpair],
        )>,
        cache: &xc_cache::ArtifactCacheContext<'_>,
        execute_primary: F,
    ) -> Result<xc_cache::ArtifactExecutionCacheResult<xc_cache::CaptureArtifact>>
    where
        F: FnMut(
            &str,
        )
            -> std::result::Result<xc_cache::CapturedDiagnostic, xc_cache::CaptureFailure>,
    {
        self.execute_with_receipt_and_reduction(retained, None, cache, execute_primary)
    }

    /// Include an explicitly budgeted reduction in the same managed receipt.
    /// The resolved record binds the capture plan and the reduction budget.
    #[cfg(feature = "hp")]
    pub fn execute_with_receipt_and_reduction<F>(
        &self,
        retained: Option<(
            &super::prefix::RetainedEvenMatrix,
            &[super::prefix::RetainedEvenEigenpair],
        )>,
        reduction: Option<&RetainedReductionRequest>,
        cache: &xc_cache::ArtifactCacheContext<'_>,
        mut execute_primary: F,
    ) -> Result<xc_cache::ArtifactExecutionCacheResult<xc_cache::CaptureArtifact>>
    where
        F: FnMut(
            &str,
        )
            -> std::result::Result<xc_cache::CapturedDiagnostic, xc_cache::CaptureFailure>,
    {
        self.validate()?;
        let expected = self.receipt()?;
        // Put the ladder before its checkpoints even when IDs sort differently.
        let mut requested = expected
            .outcomes()
            .keys()
            .filter(|id| !id.starts_with("prefix_"))
            .cloned()
            .collect::<Vec<_>>();
        if self.capture_prefix_analysis {
            requested.push("prefix_ladder".into());
            requested.extend(
                self.prefix_checkpoint_dimensions
                    .iter()
                    .map(|k| format!("prefix_checkpoint_{k}")),
            );
        }
        let resolved_plan = if let Some(budget) = reduction {
            requested.push("retained_reduction".into());
            let mut budget = budget.clone();
            budget.relative_tolerance = xc_core::DecimalLiteral::new(&budget.relative_tolerance)?
                .canonical()?
                .to_string();
            serde_json::json!({"capture":self.resolved_value()?,"retained_reduction":budget})
        } else {
            self.resolved_value()?
        };
        let expected = xc_core::CaptureReceipt::new(&resolved_plan, requested.clone())?;
        let mut prefix: Option<
            xc_cache::ArtifactExecutionCacheResult<super::prefix::CcmPrefixAnalysis>,
        > = None;
        let prefix_sources = |manifest: Option<&xc_cache::ArtifactManifest>| {
            if let Some(manifest) = manifest {
                return vec![manifest.clone()];
            }
            retained
                .map(|(matrix, eigenpairs)| {
                    std::iter::once(matrix.manifest())
                        .chain(eigenpairs.iter().map(|s| s.manifest()))
                        .cloned()
                        .collect()
                })
                .unwrap_or_default()
        };
        let record = xc_cache::capture_and_persist(
            &resolved_plan,
            requested,
            |id| {
                use xc_cache::{CaptureFailure, CapturedDiagnostic};
                if id == "retained_reduction" {
                    let Some((matrix, _)) = retained else {
                        return Err(CaptureFailure::Missing {
                            reason: "retained even matrix unavailable".into(),
                        });
                    };
                    let budget = reduction.expect("reduction ID is requested only with a budget");
                    if matrix.dimension() > budget.maximum_dimension
                        || matrix.source_precision_bits() > budget.working_precision_bits
                    {
                        return Err(CaptureFailure::Blocked {
                            reason:
                                "retained reduction exceeds dimension or source precision budget"
                                    .into(),
                        });
                    }
                    let result = super::prefix::check_retained_reduction_via_cache(
                        matrix,
                        budget.working_precision_bits,
                        budget.maximum_dimension,
                        &budget.relative_tolerance,
                        cache,
                    )
                    .map_err(CaptureFailure::failed)?;
                    let sources = result
                        .produced_manifest
                        .as_ref()
                        .or(result.reused_manifest.as_ref())
                        .map(|m| vec![m.clone()])
                        .unwrap_or_else(|| vec![matrix.manifest().clone()]);
                    return CapturedDiagnostic::new(&result.value, sources)
                        .map_err(CaptureFailure::failed);
                }
                if id == "prefix_ladder" {
                    let Some((matrix, eigenpairs)) = retained else {
                        return Err(CaptureFailure::Missing {
                            reason: "retained even matrix unavailable".into(),
                        });
                    };
                    let result = self
                        .capture_retained_diagnostics(matrix, eigenpairs, cache)
                        .map_err(CaptureFailure::failed)?
                        .ok_or_else(|| CaptureFailure::failed("prefix capture is disabled"))?;
                    let diagnostic = CapturedDiagnostic::new(
                        &result.value,
                        prefix_sources(
                            result
                                .produced_manifest
                                .as_ref()
                                .or(result.reused_manifest.as_ref()),
                        ),
                    )
                    .map_err(CaptureFailure::failed)?;
                    prefix = Some(result);
                    return Ok(diagnostic);
                }
                if let Some(dimension) = id.strip_prefix("prefix_checkpoint_") {
                    let Some(result) = &prefix else {
                        return Err(CaptureFailure::Blocked {
                            reason: "prefix ladder unavailable".into(),
                        });
                    };
                    let checkpoint = result
                        .value
                        .checkpoints
                        .iter()
                        .find(|c| c.dimension.to_string() == dimension)
                        .ok_or_else(|| CaptureFailure::Missing {
                            reason: "requested checkpoint not reached".into(),
                        })?;
                    if checkpoint.eigenpair_source.is_none() {
                        return Err(CaptureFailure::Missing { reason: "checkpoint eigenstate unavailable; partial export retained in prefix ladder".into() });
                    }
                    return CapturedDiagnostic::new(
                        checkpoint,
                        prefix_sources(
                            result
                                .produced_manifest
                                .as_ref()
                                .or(result.reused_manifest.as_ref()),
                        ),
                    )
                    .map_err(CaptureFailure::failed);
                }
                execute_primary(id)
            },
            cache,
        )?;
        record.value.receipt.validate_against(&expected)?;
        Ok(record)
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
        self.validate_for_source_precision(source_precision_bits)?;
        if !self.capture_prefix_analysis {
            return Ok(None);
        }
        let working_precision_bits = self
            .prefix_working_precision_bits
            .unwrap_or(source_precision_bits);
        let full = xc_numerics::reduction::roundtrip_decimal_digits(working_precision_bits);
        let mut widths = vec![80, 96, 112, full];
        widths.sort_unstable();
        widths.dedup();
        widths.retain(|d| *d <= full);
        if widths.is_empty() {
            widths.push(full);
        }
        // A conservative export screen chosen from arithmetic precision,
        // explicitly not an estimate of matrix-assembly accuracy.
        let digits = ((working_precision_bits.saturating_sub(48) as usize) * 3 / 10).max(1);
        let (margin, widths, tolerance) = if let Some(policy) = &self.prefix_export_policy {
            (
                policy.pivot_margin_bits,
                policy.significant_digits.clone(),
                policy.relative_tolerance.clone(),
            )
        } else {
            (32, widths, format!("1e-{digits}"))
        };
        Ok(Some(super::prefix::PrefixAnalysisOptions {
            working_precision_bits,
            pivot_margin_bits: margin,
            diagnostics: self.prefix_diagnostics.unwrap_or_default(),
            checkpoint_dimensions: self.prefix_checkpoint_dimensions.clone(),
            export_significant_digits: widths,
            export_relative_tolerance: xc_core::DecimalLiteral::new(tolerance)?
                .canonical()?
                .to_string(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prefix_diagnostic_policy_is_explicit_and_old_plans_do_not_gain_new_work() {
        let full = CcmCapturePlan::ultra(2, 3).unwrap();
        assert_eq!(
            full.prefix_diagnostics,
            Some(PrefixDiagnosticPolicy::full())
        );
        let mut old = serde_json::to_value(&full).unwrap();
        old.as_object_mut().unwrap().remove("prefix_diagnostics");
        let old: CcmCapturePlan = serde_json::from_value(old).unwrap();
        old.validate().unwrap();
        assert!(old.prefix_diagnostics.is_none());
        assert_ne!(
            full.receipt().unwrap().plan_digest(),
            old.receipt().unwrap().plan_digest()
        );
        let lean = full
            .with_prefix_diagnostics(PrefixDiagnosticPolicy::default())
            .unwrap();
        assert_eq!(lean, old);
        let base = CcmCapturePlan::resolve(CcmCaptureLevel::Claim, 0, 3).unwrap();
        let a = base
            .clone()
            .with_prefix_diagnostics(PrefixDiagnosticPolicy::full())
            .unwrap()
            .with_prefix_working_precision(512)
            .unwrap();
        let b = base
            .with_prefix_working_precision(512)
            .unwrap()
            .with_prefix_diagnostics(PrefixDiagnosticPolicy::full())
            .unwrap();
        assert_eq!(a, b);
        #[cfg(feature = "hp")]
        {
            assert!(
                a.prefix_options(256)
                    .unwrap()
                    .unwrap()
                    .diagnostics
                    .third_inverse_moment
            );
            assert!(
                !old.prefix_options(256)
                    .unwrap()
                    .unwrap()
                    .diagnostics
                    .third_inverse_moment
            );
        }
    }

    #[test]
    fn prefix_builders_commute_and_bind_precision_before_execution() {
        let policy = PrefixExportPolicy {
            pivot_margin_bits: 128,
            significant_digits: vec![80, 158],
            relative_tolerance: "0.1e-99".into(),
        };
        for level in [
            CcmCaptureLevel::Claim,
            CcmCaptureLevel::Research,
            CcmCaptureLevel::Ultra,
        ] {
            let base = CcmCapturePlan::resolve(level, 8, 33).unwrap();
            let expected = base
                .clone()
                .with_prefix_checkpoints(vec![17, 33])
                .unwrap()
                .with_prefix_working_precision(512)
                .unwrap()
                .with_prefix_export_policy(policy.clone())
                .unwrap();
            for order in [
                [0, 1, 2],
                [0, 2, 1],
                [1, 0, 2],
                [1, 2, 0],
                [2, 0, 1],
                [2, 1, 0],
            ] {
                let mut actual = base.clone();
                for operation in order {
                    actual = match operation {
                        0 => actual.with_prefix_checkpoints(vec![17, 33]),
                        1 => actual.with_prefix_working_precision(512),
                        _ => actual.with_prefix_export_policy(policy.clone()),
                    }
                    .unwrap();
                }
                assert_eq!(actual.receipt().unwrap(), expected.receipt().unwrap());
            }
            let inherited = base
                .clone()
                .with_prefix_export_policy(policy.clone())
                .unwrap();
            assert!(inherited.validate_for_source_precision(128).is_err());
            inherited.validate_for_source_precision(256).unwrap();
            assert!(inherited.with_prefix_working_precision(128).is_err());
            let mut encoded = serde_json::to_value(&expected).unwrap();
            encoded["prefix_export_policy"]["pivot_margin_bits"] = 512.into();
            let malformed: CcmCapturePlan = serde_json::from_value(encoded).unwrap();
            assert!(malformed.validate().is_err());
            assert!(malformed.receipt().is_err());
            assert!(base
                .with_prefix_export_policy(PrefixExportPolicy {
                    pivot_margin_bits: 1_000_000,
                    ..policy.clone()
                })
                .is_err());
        }
    }
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
    fn ultra_receipt_cannot_hide_missing_targets_or_the_retained_phase() {
        use xc_core::{DiagnosticOutcome, EvidenceRef};
        let plan = CcmCapturePlan::ultra(8, 33).unwrap();
        let expected = plan.receipt().unwrap();
        let mut receipt = expected.clone();
        for id in expected.outcomes().keys() {
            let outcome = if id == "deviation_decomposition" {
                DiagnosticOutcome::Missing {
                    reason: "auxiliary private target unavailable".into(),
                }
            } else if id.starts_with("prefix_") {
                continue;
            } else {
                DiagnosticOutcome::Completed {
                    evidence: vec![EvidenceRef::new("test", id, "synthetic evidence")],
                }
            };
            receipt.record(id, outcome).unwrap();
        }
        receipt.validate_against(&expected).unwrap();
        assert!(!receipt.is_complete());
        assert!(receipt.outcomes().contains_key("prefix_checkpoint_33"));
        assert!(!CcmCapturePlan::resolve(CcmCaptureLevel::Maximum, 8, 33)
            .unwrap()
            .receipt()
            .unwrap()
            .outcomes()
            .contains_key("deviation_decomposition"));
        let mut encoded = serde_json::to_value(&receipt).unwrap();
        encoded["outcomes"]
            .as_object_mut()
            .unwrap()
            .remove("prefix_checkpoint_33");
        let altered: xc_core::CaptureReceipt = serde_json::from_value(encoded).unwrap();
        assert!(altered.validate_against(&expected).is_err());
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
    fn prefix_precision_override_is_replayable_and_cannot_down_round() {
        let original = CcmCapturePlan::ultra(8, 33).unwrap();
        let old_json = serde_json::to_value(&original).unwrap();
        assert!(old_json.get("prefix_working_precision_bits").is_none());
        let plan = original.clone().with_prefix_working_precision(512).unwrap();
        assert_ne!(plan.receipt().unwrap(), original.receipt().unwrap());
        let encoded = serde_json::to_vec(&plan).unwrap();
        let restored: CcmCapturePlan = serde_json::from_slice(&encoded).unwrap();
        let options = restored.prefix_options(256).unwrap().unwrap();
        assert_eq!(options.working_precision_bits, 512);
        assert!(restored.prefix_options(1024).is_err());
        assert!(original.with_prefix_working_precision(63).is_err());
        let policy = PrefixExportPolicy {
            pivot_margin_bits: 48,
            significant_digits: vec![112, 158],
            relative_tolerance: "0.1e-99".into(),
        };
        let custom = plan.with_prefix_export_policy(policy).unwrap();
        let options = custom.prefix_options(256).unwrap().unwrap();
        assert_eq!(options.pivot_margin_bits, 48);
        assert_eq!(options.export_relative_tolerance, "1e-100");
        assert_eq!(options.export_significant_digits, vec![112, 158]);
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
