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
        "quadrature" => ("0.13.0", "0.13.0", None, SCHEMA_V1),
        "ccm-components" => ("0.13.0", "0.13.0", None, SCHEMA_V1),
        "ccm-matrices" => ("0.13.0", "0.13.0", None, SCHEMA_V1),
        "weil-states" => ("0.13.0", "0.13.0", None, SCHEMA_V1),
        "prolate" => ("0.13.0", "0.13.0", None, SCHEMA_V1),
        "ccm-roots" => ("0.13.0", "0.13.0", None, SCHEMA_V1),
        "ccm-evidence" => ("0.13.0", "0.13.0", None, SCHEMA_V1),
        // Eigenfunction profiles and target-distance measurements. Introduced
        // in 0.14.0, so its producer floor starts there rather than at 0.13.0.
        "ccm-distance" => ("0.14.0", "0.13.0", None, SCHEMA_V1),
        // Canonical protocol fixtures exercise backend-neutral mechanics with
        // synthetic families. They remain explicit rather than receiving a
        // permissive unknown-family fallback.
        "ccm" => ("0.13.0", "0.13.0", None, SCHEMA_V1),
        "fixture" => ("0.13.0", "0.13.0", None, SCHEMA_V1),
        "research_evidence" => ("0.13.0", "0.13.0", None, SCHEMA_V1),
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
        "gauss_legendre_rule" => "0.13.0",
        "quadrature_rule" => "0.13.0",
        "quadrature_reference_table" => "0.13.0",
        "quadrature_validation" => "0.13.0",
        "ccm_prime_enumeration" => "0.13.0",
        "ccm_archimedean_integrals" => "0.13.0",
        "ccm_archimedean_component" => "0.13.0",
        "ccm_prime_component" => "0.13.0",
        "ccm_pole_component" => "0.13.0",
        "ccm_tau_matrix" => "0.13.0",
        "ccm_even_sector_matrix" => "0.13.0",
        "ccm_odd_sector_matrix" => "0.13.0",
        "ccm_sector_tridiagonal" => "0.13.0",
        "ccm_sector_transform" => "0.13.0",
        "ccm_reduced_operator" => "0.13.0",
        "ccm_factorization" => "0.13.0",
        "ccm_sector_eigenvalues" => "0.13.0",
        "ccm_sector_spectrum" => "0.13.0",
        "ccm_sector_gap" => "0.13.0",
        // ccm-distance family, introduced in 0.14.0.
        "ccm_discretization_distance" => "0.14.0",
        "ccm_distance_resolution_evidence" => "0.14.1",
        "ccm_eigenfunction_profile" => "0.14.0",
        "ccm_target_distance" => "0.14.0",
        "ccm_target_residual_analysis" => "0.14.1",
        "ccm_weil_eigenpair" => "0.13.0",
        "ccm_weil_plunge_state" => "0.13.0",
        "ccm_weil_sonin_state" => "0.13.0",
        "ccm_source_eigenbasis" => "0.13.0",
        "prolate_eigenvalue_spectrum" => "0.13.0",
        "ccm_prolate_spectrum" => "0.13.0",
        "ccm_prolate_basis" => "0.13.0",
        "ccm_prolate_candidate" => "0.13.0",
        "ccm_band_concentration" => "0.13.0",
        "ccm_secular_source" => "0.13.0",
        "ccm_root_count_window" => "0.13.0",
        "ccm_root_discovery_window" => "0.13.0",
        "ccm_root_refinement" => "0.13.0",
        "ccm_spectral_window" => "0.13.0",
        "ccm_post_discovery_comparison" => "0.13.0",
        "ccm_convergence_diagnostics" => "0.13.0",
        "ccm_prefix_analysis" => "0.14.4",
        "ccm_root_conditioning_analysis" => "0.14.1",
        "ccm_deviation_decomposition" => "0.14.1",
        "ccm_prime_power_response_analysis" => "0.14.1",
        "ccm_u_flow_response_analysis" => "0.14.1",
        "ccm_sector_gap_certificate" => "0.14.4",
        "ccm_cross_check_record" => "0.13.0",
        "ccm_validation_record" => "0.13.0",
        "ccm_certificate_bundle" => "0.13.0",
        _ if matches!(family, "ccm" | "fixture" | "research_evidence") => "0.13.0",
        _ => {
            return Err(CacheError::InvalidManifest(format!(
                "artifact kind {artifact_kind:?} has no explicit compatibility policy in family {family:?}"
            )))
        }
    };
    let mut policy = artifact_family_compatibility_policy(family)?;
    policy.artifact_kind = Some(artifact_kind.to_owned());
    policy.minimum_producer_version = ToolkitVersion::parse(minimum_producer)?;
    if matches!(
        artifact_kind,
        "ccm_distance_resolution_evidence"
            | "ccm_target_residual_analysis"
            | "ccm_root_conditioning_analysis"
            | "ccm_deviation_decomposition"
            | "ccm_prime_power_response_analysis"
            | "ccm_u_flow_response_analysis"
            | "ccm_sector_gap_certificate"
    ) {
        policy.minimum_reader_version = ToolkitVersion::parse("0.14.1")?;
    }
    if matches!(
        artifact_kind,
        "ccm_prefix_analysis" | "ccm_sector_gap_certificate"
    ) {
        policy.minimum_reader_version = ToolkitVersion::parse("0.14.4")?;
    }
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
            || maximum_reader.is_some_and(|maximum| {
                self.maximum_reader_version
                    .as_ref()
                    .is_some_and(|policy_maximum| maximum > policy_maximum)
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

    #[test]
    fn every_managed_family_has_an_explicit_policy() {
        for family in [
            "quadrature",
            "ccm-components",
            "ccm-matrices",
            "weil-states",
            "prolate",
            "ccm-roots",
            "ccm-evidence",
            "ccm-distance",
        ] {
            assert_eq!(
                artifact_family_compatibility_policy(family).unwrap().family,
                family
            );
        }
        assert!(artifact_family_compatibility_policy("unregistered").is_err());
    }

    /// The `ccm-distance` family is new in 0.14.0, so its producer floor
    /// starts there: a 0.13.x toolkit never produced these kinds, and an
    /// artifact claiming otherwise is not admissible. Its reader floor stays
    /// at 0.13.0 so the family carries no reader restriction of its own.
    #[test]
    fn ccm_distance_family_floors_start_at_its_introducing_release() {
        let policy = artifact_family_compatibility_policy("ccm-distance").unwrap();
        assert_eq!(
            policy.minimum_producer_version,
            ToolkitVersion::parse("0.14.0").unwrap()
        );
        assert_eq!(
            policy.minimum_reader_version,
            ToolkitVersion::parse("0.13.0").unwrap()
        );
        for kind in [
            "ccm_discretization_distance",
            "ccm_eigenfunction_profile",
            "ccm_target_distance",
        ] {
            let kind_policy = artifact_compatibility_policy("ccm-distance", kind).unwrap();
            assert_eq!(
                kind_policy.minimum_producer_version,
                ToolkitVersion::parse("0.14.0").unwrap()
            );
            assert!(kind_policy
                .validate_manifest_versions(
                    1,
                    &ToolkitVersion::parse("0.13.5").unwrap(),
                    &ToolkitVersion::parse("0.13.0").unwrap(),
                    None,
                )
                .is_err());
        }
        let evidence =
            artifact_compatibility_policy("ccm-distance", "ccm_distance_resolution_evidence")
                .unwrap();
        assert_eq!(
            evidence.minimum_producer_version,
            ToolkitVersion::parse("0.14.1").unwrap()
        );
        assert_eq!(
            evidence.minimum_reader_version,
            ToolkitVersion::parse("0.14.1").unwrap()
        );
        assert!(evidence
            .validate_manifest_versions(
                1,
                &ToolkitVersion::parse("0.14.0").unwrap(),
                &ToolkitVersion::parse("0.14.1").unwrap(),
                None,
            )
            .is_err());
        let residual =
            artifact_compatibility_policy("ccm-distance", "ccm_target_residual_analysis").unwrap();
        assert_eq!(
            residual.minimum_producer_version,
            ToolkitVersion::parse("0.14.1").unwrap()
        );
        assert_eq!(
            residual.minimum_reader_version,
            ToolkitVersion::parse("0.14.1").unwrap()
        );
        let root_conditioning =
            artifact_compatibility_policy("ccm-evidence", "ccm_root_conditioning_analysis")
                .unwrap();
        assert_eq!(
            root_conditioning.minimum_producer_version,
            ToolkitVersion::parse("0.14.1").unwrap()
        );
        assert_eq!(
            root_conditioning.minimum_reader_version,
            ToolkitVersion::parse("0.14.1").unwrap()
        );
        let prime_response =
            artifact_compatibility_policy("ccm-evidence", "ccm_prime_power_response_analysis")
                .unwrap();
        assert_eq!(
            prime_response.minimum_producer_version,
            ToolkitVersion::parse("0.14.1").unwrap()
        );
        assert_eq!(
            prime_response.minimum_reader_version,
            ToolkitVersion::parse("0.14.1").unwrap()
        );
        let u_flow_response =
            artifact_compatibility_policy("ccm-evidence", "ccm_u_flow_response_analysis").unwrap();
        assert_eq!(
            u_flow_response.minimum_producer_version,
            ToolkitVersion::parse("0.14.1").unwrap()
        );
        assert_eq!(
            u_flow_response.minimum_reader_version,
            ToolkitVersion::parse("0.14.1").unwrap()
        );
        let sector_gap_certificate =
            artifact_compatibility_policy("ccm-evidence", "ccm_sector_gap_certificate").unwrap();
        assert_eq!(
            sector_gap_certificate.minimum_producer_version,
            ToolkitVersion::parse("0.14.4").unwrap()
        );
        assert_eq!(
            sector_gap_certificate.minimum_reader_version,
            ToolkitVersion::parse("0.14.4").unwrap()
        );
    }

    #[test]
    fn producer_below_family_floor_is_rejected_for_recomputation() {
        let policy = artifact_family_compatibility_policy("ccm-matrices").unwrap();
        let error = policy
            .validate_manifest_versions(
                1,
                &ToolkitVersion::parse("0.12.99").unwrap(),
                &ToolkitVersion::parse("0.13.0").unwrap(),
                None,
            )
            .unwrap_err();
        assert!(error.to_string().contains("precedes"));
    }

    #[test]
    fn current_ccm_artifact_floors_match_the_release() {
        let discovery =
            artifact_compatibility_policy("ccm-roots", "ccm_root_discovery_window").unwrap();
        let refinement = artifact_compatibility_policy("ccm-roots", "ccm_root_refinement").unwrap();
        let tau = artifact_compatibility_policy("ccm-matrices", "ccm_tau_matrix").unwrap();
        let tridiagonal =
            artifact_compatibility_policy("ccm-matrices", "ccm_sector_tridiagonal").unwrap();
        let transform =
            artifact_compatibility_policy("ccm-matrices", "ccm_sector_transform").unwrap();
        let sector_eigenvalues =
            artifact_compatibility_policy("weil-states", "ccm_sector_eigenvalues").unwrap();
        let eigenpair = artifact_compatibility_policy("weil-states", "ccm_weil_eigenpair").unwrap();
        assert_eq!(
            discovery.minimum_producer_version,
            ToolkitVersion::parse("0.13.0").unwrap()
        );
        assert_eq!(
            refinement.minimum_producer_version,
            ToolkitVersion::parse("0.13.0").unwrap()
        );
        assert_eq!(
            eigenpair.minimum_producer_version,
            ToolkitVersion::parse("0.13.0").unwrap()
        );
        assert_eq!(
            tau.minimum_producer_version,
            ToolkitVersion::parse("0.13.0").unwrap()
        );
        assert_eq!(
            tridiagonal.minimum_producer_version,
            ToolkitVersion::parse("0.13.0").unwrap()
        );
        assert_eq!(
            transform.minimum_producer_version,
            ToolkitVersion::parse("0.13.0").unwrap()
        );
        assert_eq!(
            sector_eigenvalues.minimum_producer_version,
            ToolkitVersion::parse("0.13.0").unwrap()
        );
    }
}
