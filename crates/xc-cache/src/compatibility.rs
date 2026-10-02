use crate::{CacheError, ToolkitVersion};

/// Central compatibility floor for one canonical cache artifact family.
///
/// Raising `minimum_producer_version` invalidates older results without
/// deleting immutable objects: consumers treat them as misses and recompute.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactFamilyCompatibilityPolicy {
    pub family: String,
    pub artifact_kind: Option<String>,
    pub minimum_producer_version: ToolkitVersion,
    pub minimum_reader_version: ToolkitVersion,
    pub maximum_reader_version: Option<ToolkitVersion>,
    pub accepted_manifest_schema_versions: &'static [u32],
}

const SCHEMA_V1: &[u32] = &[1];

/// Producer and reader floor for every managed family and kind. Toolkit 0.16.0
/// started the artifact repositories from a clean slate: artifacts produced by
/// earlier releases are misses and are recomputed, and earlier readers refuse
/// 0.16.0 artifacts instead of misreading their changed formats.
pub const CLEAN_SLATE: &str = "0.16.0";

pub fn current_toolkit_version() -> Result<ToolkitVersion, CacheError> {
    ToolkitVersion::parse(env!("CARGO_PKG_VERSION"))
}

/// Return the release policy for a managed cache family.
///
/// Families use the current canonical baseline unless an explicit override is
/// added here. This gives new typed families a safe floor without duplicating
/// version constants throughout numerical implementations.
pub fn artifact_family_compatibility_policy(
    family: &str,
) -> Result<ArtifactFamilyCompatibilityPolicy, CacheError> {
    // Keep every family in its own arm even while floors coincide. A defect in
    // one family can then raise only that producer floor in a patch release.
    let (minimum_producer, minimum_reader, maximum_reader, schemas) = match family {
        "quadrature" => (CLEAN_SLATE, CLEAN_SLATE, None, SCHEMA_V1),
        "ccm-components" => (CLEAN_SLATE, CLEAN_SLATE, None, SCHEMA_V1),
        "ccm-matrices" => (CLEAN_SLATE, CLEAN_SLATE, None, SCHEMA_V1),
        "weil-states" => (CLEAN_SLATE, CLEAN_SLATE, None, SCHEMA_V1),
        "prolate" => (CLEAN_SLATE, CLEAN_SLATE, None, SCHEMA_V1),
        // Reserved schema support; numerical producers remain separately opt-in.
        "maynard-tao" => (CLEAN_SLATE, CLEAN_SLATE, None, SCHEMA_V1),
        "ccm-roots" => (CLEAN_SLATE, CLEAN_SLATE, None, SCHEMA_V1),
        "ccm-evidence" => (CLEAN_SLATE, CLEAN_SLATE, None, SCHEMA_V1),
        // Eigenfunction profiles and target-distance measurements.
        "ccm-distance" => (CLEAN_SLATE, CLEAN_SLATE, None, SCHEMA_V1),
        // Canonical protocol fixtures exercise backend-neutral mechanics with
        // synthetic families. They remain explicit rather than receiving a
        // permissive unknown-family fallback.
        "ccm" => (CLEAN_SLATE, CLEAN_SLATE, None, SCHEMA_V1),
        "fixture" => (CLEAN_SLATE, CLEAN_SLATE, None, SCHEMA_V1),
        "research_evidence" => (CLEAN_SLATE, CLEAN_SLATE, None, SCHEMA_V1),
        _ => {
            return Err(CacheError::InvalidManifest(format!(
                "artifact family {family:?} has no explicit compatibility policy"
            )))
        }
    };
    Ok(ArtifactFamilyCompatibilityPolicy {
        family: family.to_owned(),
        artifact_kind: None,
        minimum_producer_version: ToolkitVersion::parse(minimum_producer)?,
        minimum_reader_version: ToolkitVersion::parse(minimum_reader)?,
        maximum_reader_version: maximum_reader.map(ToolkitVersion::parse).transpose()?,
        accepted_manifest_schema_versions: schemas,
    })
}

/// Resolve compatibility for one granular artifact kind. Each kind is listed
/// explicitly so a patch release can raise (for example) only the Tau floor.
pub fn artifact_compatibility_policy(
    family: &str,
    artifact_kind: &str,
) -> Result<ArtifactFamilyCompatibilityPolicy, CacheError> {
    let minimum_producer = match artifact_kind {
        "maynard_basis" | "maynard_moment_table" | "maynard_operator" | "maynard_candidate" | "maynard_bound" | "maynard_certificate" => CLEAN_SLATE,
        "gauss_legendre_rule" => CLEAN_SLATE,
        "quadrature_rule" => CLEAN_SLATE,
        "quadrature_reference_table" => CLEAN_SLATE,
        "quadrature_validation" => CLEAN_SLATE,
        "ccm_prime_enumeration" => CLEAN_SLATE,
        "ccm_archimedean_integrals" => CLEAN_SLATE,
        "ccm_archimedean_component" => CLEAN_SLATE,
        "ccm_prime_component" => CLEAN_SLATE,
        "ccm_pole_component" => CLEAN_SLATE,
        "ccm_tau_matrix" => CLEAN_SLATE,
        "ccm_even_sector_matrix" => CLEAN_SLATE,
        "ccm_odd_sector_matrix" => CLEAN_SLATE,
        "ccm_sector_tridiagonal" => CLEAN_SLATE,
        "ccm_sector_transform" => CLEAN_SLATE,
        "ccm_reduced_operator" => CLEAN_SLATE,
        "ccm_factorization" => CLEAN_SLATE,
        "ccm_sector_eigenvalues" => CLEAN_SLATE,
        "ccm_sector_spectrum" => CLEAN_SLATE,
        "ccm_sector_gap" => CLEAN_SLATE,
        // ccm-distance family.
        "ccm_discretization_distance" => CLEAN_SLATE,
        "ccm_distance_resolution_evidence" => CLEAN_SLATE,
        "ccm_eigenfunction_profile" => CLEAN_SLATE,
        "ccm_target_distance" => CLEAN_SLATE,
        "ccm_target_residual_analysis" => CLEAN_SLATE,
        "ccm_target_comparison_analysis" => CLEAN_SLATE,
        "ccm_weil_eigenpair" => CLEAN_SLATE,
        "ccm_weil_plunge_state" => CLEAN_SLATE,
        "ccm_weil_sonin_state" => CLEAN_SLATE,
        "ccm_source_eigenbasis" => CLEAN_SLATE,
        "prolate_eigenvalue_spectrum" => CLEAN_SLATE,
        "ccm_prolate_spectrum" => CLEAN_SLATE,
        "ccm_prolate_basis" => CLEAN_SLATE,
        "ccm_prolate_candidate" => CLEAN_SLATE,
        "ccm_band_concentration" => CLEAN_SLATE,
        "ccm_secular_source" => CLEAN_SLATE,
        "ccm_root_count_window" => CLEAN_SLATE,
        "ccm_root_discovery_window" => CLEAN_SLATE,
        "ccm_root_refinement" => CLEAN_SLATE,
        "ccm_spectral_window" => CLEAN_SLATE,
        "ccm_post_discovery_comparison" => CLEAN_SLATE,
        "ccm_convergence_diagnostics" => CLEAN_SLATE,
        "ccm_prefix_analysis" => CLEAN_SLATE,
        "ccm_retained_reduction_check" => CLEAN_SLATE,
        "ccm_state_geometry_analysis" | "research_observation_packet" | "research_reference_dataset"
        | "ccm_reference_source" | "ccm_reference_projection_analysis" | "ccm_indexed_transform_analysis"
        | "ccm_operator_energy_analysis" | "ccm_root_band_analysis" | "ccm_stabilization_analysis" => CLEAN_SLATE,
        "ccm_compactness_analysis" | "ccm_arithmetic_energy_analysis" | "ccm_directional_response_analysis" | "ccm_weighted_tail_analysis" | "ccm_spectral_cluster_analysis" | "ccm_resolution_budget_analysis" | "ccm_energy_allowance_analysis" | "ccm_weighted_reference_projection" | "ccm_signed_transform_analysis" | "ccm_external_research_source" | "ccm_complex_transform_analysis" | "ccm_root_transport_analysis" | "ccm_operator_cluster_analysis" | "ccm_finite_section_transfer" | "ccm_tail_operator_analysis" | "ccm_observable_budget_analysis" | "ccm_capture_preflight" | "ccm_consistency_analysis" | "ccm_configuration_comparison" | "ccm_band_reconstruction" | "ccm_transform_enclosure" => CLEAN_SLATE,
        "research_capture_receipt" | "research_hypothesis_evaluation" => CLEAN_SLATE,
        "ccm_root_conditioning_analysis" => CLEAN_SLATE,
        "ccm_deviation_decomposition" => CLEAN_SLATE,
        "ccm_prime_power_response_analysis" => CLEAN_SLATE,
        "ccm_u_flow_response_analysis" => CLEAN_SLATE,
        "ccm_sector_gap_certificate" => CLEAN_SLATE,
        "ccm_cross_check_record" => CLEAN_SLATE,
        "ccm_validation_record" => CLEAN_SLATE,
        "ccm_certificate_bundle" => CLEAN_SLATE,
        "ccm_root_certification_report" => CLEAN_SLATE,
        "ccm_assembly_error_analysis" => CLEAN_SLATE,
        "ccm_checkpoint_spectra" => CLEAN_SLATE,
        _ if matches!(family, "ccm" | "fixture" | "research_evidence") => CLEAN_SLATE,
        _ => {
            return Err(CacheError::InvalidManifest(format!(
                "artifact kind {artifact_kind:?} has no explicit compatibility policy in family {family:?}"
            )))
        }
    };
    let mut policy = artifact_family_compatibility_policy(family)?;
    policy.artifact_kind = Some(artifact_kind.to_owned());
    policy.minimum_producer_version = ToolkitVersion::parse(minimum_producer)?;
    Ok(policy)
}

impl ArtifactFamilyCompatibilityPolicy {
    pub fn validate_manifest_versions(
        &self,
        schema_version: u32,
        producer: &ToolkitVersion,
        minimum_reader: &ToolkitVersion,
        maximum_reader: Option<&ToolkitVersion>,
    ) -> Result<(), CacheError> {
        for version in [
            producer,
            minimum_reader,
            &self.minimum_producer_version,
            &self.minimum_reader_version,
        ]
        .into_iter()
        .chain(maximum_reader)
        .chain(self.maximum_reader_version.as_ref())
        {
            version.validate()?;
        }
        if maximum_reader.is_some_and(|maximum| maximum < minimum_reader)
            || self
                .maximum_reader_version
                .as_ref()
                .is_some_and(|maximum| maximum < &self.minimum_reader_version)
        {
            return Err(CacheError::InvalidManifest(
                "reader compatibility window is reversed".into(),
            ));
        }
        if !self
            .accepted_manifest_schema_versions
            .contains(&schema_version)
        {
            return Err(CacheError::InvalidManifest(format!(
                "schema version {schema_version} is not accepted for family {:?}",
                self.family
            )));
        }
        if producer < &self.minimum_producer_version {
            return Err(CacheError::InvalidManifest(format!(
                "producer toolkit {producer} precedes the {:?} family floor {}",
                self.family, self.minimum_producer_version
            )));
        }
        if minimum_reader < &self.minimum_reader_version
            || self
                .maximum_reader_version
                .as_ref()
                .is_some_and(|policy_maximum| {
                    maximum_reader.is_none_or(|maximum| maximum > policy_maximum)
                })
        {
            return Err(CacheError::InvalidManifest(format!(
                "reader compatibility is outside the {:?} family policy",
                self.family
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MANAGED_FAMILIES: [&str; 9] = [
        "quadrature",
        "ccm-components",
        "ccm-matrices",
        "weil-states",
        "prolate",
        "maynard-tao",
        "ccm-roots",
        "ccm-evidence",
        "ccm-distance",
    ];

    fn version(text: &str) -> ToolkitVersion {
        ToolkitVersion::parse(text).unwrap()
    }

    #[test]
    fn every_managed_family_has_an_explicit_policy() {
        for family in MANAGED_FAMILIES {
            assert_eq!(
                artifact_family_compatibility_policy(family).unwrap().family,
                family
            );
        }
        assert!(artifact_family_compatibility_policy("unregistered").is_err());
        assert!(artifact_compatibility_policy("ccm-evidence", "unregistered_kind").is_err());
    }

    /// Toolkit 0.16.0 started the artifact repositories from a clean slate:
    /// every registered kind rejects earlier producers and earlier reader
    /// floors, and accepts its own release.
    #[test]
    fn every_registered_kind_rejects_artifacts_before_the_clean_slate() {
        let floor = version(CLEAN_SLATE);
        assert!(floor <= current_toolkit_version().unwrap());
        for family in MANAGED_FAMILIES {
            let kinds = crate::artifact_kinds_for_family(family).unwrap();
            assert!(!kinds.is_empty(), "{family}");
            for kind in kinds {
                let policy = artifact_compatibility_policy(family, kind).unwrap();
                assert_eq!(policy.minimum_producer_version, floor, "{family}/{kind}");
                assert_eq!(policy.minimum_reader_version, floor, "{family}/{kind}");
                assert!(policy
                    .validate_manifest_versions(1, &floor, &floor, None)
                    .is_ok());
                let earlier_producer = policy
                    .validate_manifest_versions(1, &version("0.15.2"), &floor, None)
                    .unwrap_err();
                assert!(earlier_producer.to_string().contains("precedes"));
                assert!(policy
                    .validate_manifest_versions(1, &floor, &version("0.15.2"), None)
                    .is_err());
            }
        }
    }
}
