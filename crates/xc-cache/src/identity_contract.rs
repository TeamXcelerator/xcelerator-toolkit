//! Typed numerical requests and the managed kind contract catalog.
//!
//! Existing compliant semantic builders keep their historical identities.
//! Converted builders opt into a per-kind revision through `NumericalRequest`;
//! release, execution and disclosure information has no field in this API.
use crate::{CacheError, ContentDigest, SemanticKeyEnvelope};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const MANAGED_FAMILIES: [&str; 9] = [
    "quadrature",
    "ccm-components",
    "ccm-matrices",
    "weil-states",
    "prolate",
    "ccm-roots",
    "ccm-evidence",
    "ccm-distance",
    "maynard-tao",
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactIdentityContract {
    pub family: &'static str,
    pub kind: &'static str,
    pub value_revision: u32,
    pub payload_codec: &'static str,
    /// Owning typed producer module; None means registration/legacy decoding only.
    pub producer: Option<&'static str>,
    pub private_only: bool,
    pub reserved: bool,
}

/// Exhaustive reviewed contracts. Adding a routing registration alone cannot
/// activate a producer: it must also declare its codec and owner here.
pub fn artifact_identity_contract(kind: &str) -> Result<ArtifactIdentityContract, CacheError> {
    let (family, kind, payload_codec, producer, reserved) = match kind {
        "gauss_legendre_rule" => (
            "quadrature",
            "gauss_legendre_rule",
            "gauss-legendre-rule-v1",
            Some("xc-numerics/quadrature"),
            false,
        ),
        "quadrature_rule" => (
            "quadrature",
            "quadrature_rule",
            "quadrature-rule-v1",
            Some("xc-numerics/quadrature"),
            false,
        ),
        "quadrature_reference_table" => (
            "quadrature",
            "quadrature_reference_table",
            "registration-only",
            None,
            false,
        ),
        "quadrature_validation" => (
            "quadrature",
            "quadrature_validation",
            "registration-only",
            None,
            false,
        ),
        "ccm_prime_enumeration" => (
            "ccm-components",
            "ccm_prime_enumeration",
            "registration-only",
            None,
            false,
        ),
        "ccm_archimedean_integrals" => (
            "ccm-components",
            "ccm_archimedean_integrals",
            "portable-archimedean-integrals",
            Some("xc-spectral/ccm/hp"),
            false,
        ),
        "ccm_archimedean_component" => (
            "ccm-components",
            "ccm_archimedean_component",
            "registration-only",
            None,
            false,
        ),
        "ccm_prime_component" => (
            "ccm-components",
            "ccm_prime_component",
            "portable-prime-component",
            Some("xc-spectral/ccm/hp"),
            false,
        ),
        "ccm_pole_component" => (
            "ccm-components",
            "ccm_pole_component",
            "registration-only",
            None,
            false,
        ),
        "ccm_tau_matrix" => (
            "ccm-matrices",
            "ccm_tau_matrix",
            "portable-tau-matrix-v2",
            Some("xc-spectral/ccm/hp"),
            false,
        ),
        "ccm_even_sector_matrix" => (
            "ccm-matrices",
            "ccm_even_sector_matrix",
            "portable-sector-matrix",
            Some("xc-spectral/ccm/hp"),
            false,
        ),
        "ccm_odd_sector_matrix" => (
            "ccm-matrices",
            "ccm_odd_sector_matrix",
            "portable-sector-matrix",
            Some("xc-spectral/ccm/hp"),
            false,
        ),
        "ccm_sector_tridiagonal" => (
            "ccm-matrices",
            "ccm_sector_tridiagonal",
            "portable-tridiagonal",
            Some("xc-spectral/ccm/hp"),
            false,
        ),
        "ccm_sector_transform" => (
            "ccm-matrices",
            "ccm_sector_transform",
            "portable-sector-transform",
            Some("xc-spectral/ccm/hp"),
            false,
        ),
        "ccm_reduced_operator" => (
            "ccm-matrices",
            "ccm_reduced_operator",
            "registration-only",
            None,
            false,
        ),
        "ccm_factorization" => (
            "ccm-matrices",
            "ccm_factorization",
            "portable-factorization",
            Some("xc-spectral/ccm/hp"),
            false,
        ),
        "ccm_weil_eigenpair" => (
            "weil-states",
            "ccm_weil_eigenpair",
            "portable-weil-eigenpair",
            Some("xc-spectral/ccm/hp"),
            false,
        ),
        "ccm_sector_eigenvalues" => (
            "weil-states",
            "ccm_sector_eigenvalues",
            "portable-sector-eigenvalues",
            Some("xc-spectral/ccm/hp"),
            false,
        ),
        "ccm_sector_spectrum" => (
            "weil-states",
            "ccm_sector_spectrum",
            "portable-sector-spectrum",
            Some("xc-spectral/ccm/hp"),
            false,
        ),
        "ccm_weil_plunge_state" => (
            "weil-states",
            "ccm_weil_plunge_state",
            "registration-only",
            None,
            false,
        ),
        "ccm_weil_sonin_state" => (
            "weil-states",
            "ccm_weil_sonin_state",
            "registration-only",
            None,
            false,
        ),
        "ccm_source_eigenbasis" => (
            "weil-states",
            "ccm_source_eigenbasis",
            "registration-only",
            None,
            false,
        ),
        "prolate_eigenvalue_spectrum" => (
            "prolate",
            "prolate_eigenvalue_spectrum",
            "portable-prolate-spectrum",
            Some("xc-spectral/prolate"),
            false,
        ),
        "prolate_finite_dirichlet_modes" => (
            "prolate",
            "prolate_finite_dirichlet_modes",
            "portable-dirichlet-modes",
            Some("xc-spectral/prolate"),
            false,
        ),
        "prolate_finite_dirichlet_profile" => (
            "prolate",
            "prolate_finite_dirichlet_profile",
            "portable-dirichlet-profile",
            Some("xc-spectral/prolate"),
            false,
        ),
        "ccm_prolate_spectrum" => (
            "prolate",
            "ccm_prolate_spectrum",
            "registration-only",
            None,
            false,
        ),
        "ccm_prolate_basis" => (
            "prolate",
            "ccm_prolate_basis",
            "registration-only",
            None,
            false,
        ),
        "ccm_prolate_candidate" => (
            "prolate",
            "ccm_prolate_candidate",
            "registration-only",
            None,
            false,
        ),
        "ccm_band_concentration" => (
            "prolate",
            "ccm_band_concentration",
            "registration-only",
            None,
            false,
        ),
        "ccm_reference_source" => (
            "prolate",
            "ccm_reference_source",
            "retained-research-v2",
            Some("xc-spectral/ccm/retained_evidence"),
            false,
        ),
        "ccm_external_research_source" => (
            "prolate",
            "ccm_external_research_source",
            "retained-research-v2",
            Some("xc-spectral/ccm/retained_evidence"),
            false,
        ),
        "ccm_secular_source" => (
            "ccm-roots",
            "ccm_secular_source",
            "portable-secular-source-v2",
            Some("xc-spectral/ccm/hp"),
            false,
        ),
        "ccm_root_count_window" => (
            "ccm-roots",
            "ccm_root_count_window",
            "registration-only",
            None,
            false,
        ),
        "ccm_root_discovery_window" => (
            "ccm-roots",
            "ccm_root_discovery_window",
            "portable-root-range-v5-v6",
            Some("xc-spectral/ccm/hp"),
            false,
        ),
        "ccm_root_refinement" => (
            "ccm-roots",
            "ccm_root_refinement",
            "portable-root-range-v7-v8",
            Some("xc-spectral/ccm/hp"),
            false,
        ),
        "ccm_spectral_window" => (
            "ccm-roots",
            "ccm_spectral_window",
            "registration-only",
            None,
            false,
        ),
        "ccm_compactness_analysis" => (
            "ccm-evidence",
            "ccm_compactness_analysis",
            "retained-research-v2",
            Some("xc-spectral/ccm/retained_evidence"),
            false,
        ),
        "ccm_arithmetic_energy_analysis" => (
            "ccm-evidence",
            "ccm_arithmetic_energy_analysis",
            "retained-research-v2",
            Some("xc-spectral/ccm/retained_evidence"),
            false,
        ),
        "ccm_directional_response_analysis" => (
            "ccm-evidence",
            "ccm_directional_response_analysis",
            "retained-research-v2",
            Some("xc-spectral/ccm/retained_evidence"),
            false,
        ),
        "ccm_weighted_tail_analysis" => (
            "ccm-evidence",
            "ccm_weighted_tail_analysis",
            "retained-research-v2",
            Some("xc-spectral/ccm/retained_evidence"),
            false,
        ),
        "ccm_spectral_cluster_analysis" => (
            "ccm-evidence",
            "ccm_spectral_cluster_analysis",
            "retained-research-v2",
            Some("xc-spectral/ccm/retained_evidence"),
            false,
        ),
        "ccm_resolution_budget_analysis" => (
            "ccm-evidence",
            "ccm_resolution_budget_analysis",
            "retained-research-v2",
            Some("xc-spectral/ccm/retained_evidence"),
            false,
        ),
        "ccm_energy_allowance_analysis" => (
            "ccm-evidence",
            "ccm_energy_allowance_analysis",
            "retained-research-v2",
            Some("xc-spectral/ccm/retained_evidence"),
            false,
        ),
        "ccm_complex_transform_analysis" => (
            "ccm-evidence",
            "ccm_complex_transform_analysis",
            "retained-research-v2",
            Some("xc-spectral/ccm/retained_evidence"),
            false,
        ),
        "ccm_root_transport_analysis" => (
            "ccm-evidence",
            "ccm_root_transport_analysis",
            "retained-research-v2",
            Some("xc-spectral/ccm/retained_evidence"),
            false,
        ),
        "ccm_operator_cluster_analysis" => (
            "ccm-evidence",
            "ccm_operator_cluster_analysis",
            "retained-research-v2",
            Some("xc-spectral/ccm/retained_evidence"),
            false,
        ),
        "ccm_finite_section_transfer" => (
            "ccm-evidence",
            "ccm_finite_section_transfer",
            "retained-research-v2",
            Some("xc-spectral/ccm/retained_evidence"),
            false,
        ),
        "ccm_tail_operator_analysis" => (
            "ccm-evidence",
            "ccm_tail_operator_analysis",
            "retained-research-v2",
            Some("xc-spectral/ccm/retained_evidence"),
            false,
        ),
        "ccm_observable_budget_analysis" => (
            "ccm-evidence",
            "ccm_observable_budget_analysis",
            "retained-research-v2",
            Some("xc-spectral/ccm/retained_evidence"),
            false,
        ),
        "ccm_capture_preflight" => (
            "ccm-evidence",
            "ccm_capture_preflight",
            "retained-research-v2",
            Some("xc-spectral/ccm/retained_evidence"),
            false,
        ),
        "ccm_consistency_analysis" => (
            "ccm-evidence",
            "ccm_consistency_analysis",
            "retained-research-v2",
            Some("xc-spectral/ccm/retained_evidence"),
            false,
        ),
        "ccm_configuration_comparison" => (
            "ccm-evidence",
            "ccm_configuration_comparison",
            "retained-research-v2",
            Some("xc-spectral/ccm/retained_evidence"),
            false,
        ),
        "ccm_band_reconstruction" => (
            "ccm-evidence",
            "ccm_band_reconstruction",
            "retained-research-v2",
            Some("xc-spectral/ccm/retained_evidence"),
            false,
        ),
        "ccm_finite_diagnostic_analysis" => (
            "ccm-evidence",
            "ccm_finite_diagnostic_analysis",
            "retained-research-v2",
            Some("xc-spectral/ccm/retained_evidence"),
            false,
        ),
        "ccm_transform_enclosure" => (
            "ccm-evidence",
            "ccm_transform_enclosure",
            "retained-research-v2",
            Some("xc-spectral/ccm/retained_evidence"),
            false,
        ),
        "research_capture_receipt" => (
            "ccm-evidence",
            "research_capture_receipt",
            "research-capture-receipt-v1",
            Some("xc-cache/research_artifacts"),
            false,
        ),
        "research_hypothesis_evaluation" => (
            "ccm-evidence",
            "research_hypothesis_evaluation",
            "hypothesis-evaluation-v1",
            Some("xc-cache/research_artifacts"),
            false,
        ),
        "ccm_prefix_analysis" => (
            "ccm-evidence",
            "ccm_prefix_analysis",
            "prefix-analysis-v1",
            Some("xc-spectral/ccm"),
            false,
        ),
        "ccm_retained_reduction_check" => (
            "ccm-evidence",
            "ccm_retained_reduction_check",
            "retained-reduction-check-v1",
            Some("xc-spectral/ccm"),
            false,
        ),
        "ccm_state_geometry_analysis" => (
            "ccm-evidence",
            "ccm_state_geometry_analysis",
            "state-geometry-analysis-v1",
            Some("xc-spectral/ccm"),
            false,
        ),
        "research_observation_packet" => (
            "ccm-evidence",
            "research_observation_packet",
            "retained-research-v2",
            Some("xc-spectral/ccm/retained_evidence"),
            false,
        ),
        "research_reference_dataset" => (
            "ccm-evidence",
            "research_reference_dataset",
            "retained-research-v2",
            Some("xc-spectral/ccm/retained_evidence"),
            false,
        ),
        "ccm_indexed_transform_analysis" => (
            "ccm-evidence",
            "ccm_indexed_transform_analysis",
            "retained-research-v2",
            Some("xc-spectral/ccm/retained_evidence"),
            false,
        ),
        "ccm_operator_energy_analysis" => (
            "ccm-evidence",
            "ccm_operator_energy_analysis",
            "retained-research-v2",
            Some("xc-spectral/ccm/retained_evidence"),
            false,
        ),
        "ccm_root_band_analysis" => (
            "ccm-evidence",
            "ccm_root_band_analysis",
            "retained-research-v2",
            Some("xc-spectral/ccm/retained_evidence"),
            false,
        ),
        "ccm_stabilization_analysis" => (
            "ccm-evidence",
            "ccm_stabilization_analysis",
            "retained-research-v2",
            Some("xc-spectral/ccm/retained_evidence"),
            false,
        ),
        "ccm_convergence_diagnostics" => (
            "ccm-evidence",
            "ccm_convergence_diagnostics",
            "portable-convergence-diagnostics",
            Some("xc-spectral/ccm"),
            false,
        ),
        "ccm_root_conditioning_analysis" => (
            "ccm-evidence",
            "ccm_root_conditioning_analysis",
            "portable-root-conditioning",
            Some("xc-spectral/ccm"),
            false,
        ),
        "ccm_prime_power_response_analysis" => (
            "ccm-evidence",
            "ccm_prime_power_response_analysis",
            "portable-prime-response",
            Some("xc-spectral/ccm"),
            false,
        ),
        "ccm_u_flow_response_analysis" => (
            "ccm-evidence",
            "ccm_u_flow_response_analysis",
            "portable-u-flow-response",
            Some("xc-spectral/ccm"),
            false,
        ),
        "ccm_sector_gap" => (
            "ccm-evidence",
            "ccm_sector_gap",
            "portable-sector-gap",
            Some("xc-spectral/ccm"),
            false,
        ),
        "ccm_sector_gap_certificate" => (
            "ccm-evidence",
            "ccm_sector_gap_certificate",
            "dense-or-compact-sector-gap-certificate",
            Some("xc-spectral/ccm"),
            false,
        ),
        "ccm_post_discovery_comparison" => (
            "ccm-evidence",
            "ccm_post_discovery_comparison",
            "registration-only",
            None,
            false,
        ),
        "ccm_cross_check_record" => (
            "ccm-evidence",
            "ccm_cross_check_record",
            "registration-only",
            None,
            false,
        ),
        "ccm_validation_record" => (
            "ccm-evidence",
            "ccm_validation_record",
            "validation-record",
            Some("xc-spectral/ccm"),
            false,
        ),
        "ccm_certificate_bundle" => (
            "ccm-evidence",
            "ccm_certificate_bundle",
            "portable-certificate-bundle",
            Some("xc-spectral/ccm"),
            false,
        ),
        "ccm_root_certification_report" => (
            "ccm-evidence",
            "ccm_root_certification_report",
            "portable-root-certification-report",
            Some("xc-spectral/ccm"),
            false,
        ),
        "ccm_assembly_error_analysis" => (
            "ccm-evidence",
            "ccm_assembly_error_analysis",
            "assembly-error-analysis",
            Some("xc-spectral/ccm"),
            false,
        ),
        "ccm_checkpoint_spectra" => (
            "ccm-evidence",
            "ccm_checkpoint_spectra",
            "checkpoint-spectra",
            Some("xc-spectral/ccm"),
            false,
        ),
        "ccm_weighted_reference_projection" => (
            "ccm-distance",
            "ccm_weighted_reference_projection",
            "retained-research-v2",
            Some("xc-spectral/ccm/retained_evidence"),
            false,
        ),
        "ccm_signed_transform_analysis" => (
            "ccm-distance",
            "ccm_signed_transform_analysis",
            "retained-research-v2",
            Some("xc-spectral/ccm/retained_evidence"),
            false,
        ),
        "ccm_reference_projection_analysis" => (
            "ccm-distance",
            "ccm_reference_projection_analysis",
            "retained-research-v2",
            Some("xc-spectral/ccm/retained_evidence"),
            false,
        ),
        "ccm_deviation_decomposition" => (
            "ccm-distance",
            "ccm_deviation_decomposition",
            "portable-deviation-decomposition",
            Some("xc-spectral/distance"),
            false,
        ),
        "ccm_discretization_distance" => (
            "ccm-distance",
            "ccm_discretization_distance",
            "portable-discretization-distance",
            Some("xc-spectral/distance"),
            false,
        ),
        "ccm_distance_resolution_evidence" => (
            "ccm-distance",
            "ccm_distance_resolution_evidence",
            "portable-distance-resolution",
            Some("xc-spectral/distance"),
            false,
        ),
        "ccm_eigenfunction_profile" => (
            "ccm-distance",
            "ccm_eigenfunction_profile",
            "portable-eigenfunction-profile",
            Some("xc-spectral/distance"),
            false,
        ),
        "ccm_target_distance" => (
            "ccm-distance",
            "ccm_target_distance",
            "portable-target-distance",
            Some("xc-spectral/distance"),
            false,
        ),
        "ccm_target_residual_analysis" => (
            "ccm-distance",
            "ccm_target_residual_analysis",
            "portable-target-residual",
            Some("xc-spectral/distance"),
            false,
        ),
        "ccm_target_comparison_analysis" => (
            "ccm-distance",
            "ccm_target_comparison_analysis",
            "portable-target-comparison",
            Some("xc-spectral/distance"),
            false,
        ),
        "maynard_basis" => ("maynard-tao", "maynard_basis", "reserved", None, true),
        "maynard_moment_table" => (
            "maynard-tao",
            "maynard_moment_table",
            "reserved",
            None,
            true,
        ),
        "maynard_operator" => ("maynard-tao", "maynard_operator", "reserved", None, true),
        "maynard_candidate" => ("maynard-tao", "maynard_candidate", "reserved", None, true),
        "maynard_bound" => ("maynard-tao", "maynard_bound", "reserved", None, true),
        "maynard_certificate" => ("maynard-tao", "maynard_certificate", "reserved", None, true),
        _ => {
            return Err(CacheError::InvalidManifest(
                "artifact has no registered identity contract".into(),
            ))
        }
    };
    Ok(ArtifactIdentityContract {
        family,
        kind,
        value_revision: 1,
        payload_codec,
        producer,
        private_only: crate::artifact_kind_is_private_only(kind),
        reserved,
    })
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericalInputIdentity {
    /// Ordered role, for example left/right operand or retained eigenstate.
    pub role: String,
    pub kind: String,
    /// Meaning/codec of the exact consumed values, independent of their producer.
    pub meaning: String,
    pub content_digest: ContentDigest,
}

/// The consumed immutable payload, without its store key or producer identity.
/// Callers still admit the exact source manifest and retain it as provenance.
pub fn numerical_source_identity(
    manifest: &crate::ArtifactManifest,
    role: impl Into<String>,
    meaning: impl Into<String>,
) -> NumericalInputIdentity {
    NumericalInputIdentity {
        role: role.into(),
        kind: manifest.key.kind.clone(),
        meaning: meaning.into(),
        content_digest: manifest.content_digest.clone(),
    }
}

/// Implement on an explicit per-kind numerical parameter type. Do not
/// implement on an execution configuration or arbitrary JSON value.
pub trait NumericalRequest: Serialize {
    fn artifact_kind(&self) -> &'static str;
    fn validate_numerical_parameters(&self) -> Result<(), CacheError>;
    fn normalization(&self) -> Option<String> {
        None
    }
    fn subspace(&self) -> Option<String> {
        None
    }
}

pub fn numerical_request_key<T: NumericalRequest>(
    parameters: &T,
    inputs: &[NumericalInputIdentity],
) -> Result<SemanticKeyEnvelope, CacheError> {
    parameters.validate_numerical_parameters()?;
    let contract = artifact_identity_contract(parameters.artifact_kind())?;
    if contract.producer.is_none() {
        return Err(CacheError::InvalidManifest(
            "kind has no active numerical producer".into(),
        ));
    }
    let mut roles = BTreeSet::new();
    for input in inputs {
        if input.role.trim().is_empty()
            || input.kind.trim().is_empty()
            || input.meaning.trim().is_empty()
            || !input.content_digest.validate()
            || !roles.insert(&input.role)
        {
            return Err(CacheError::InvalidManifest(
                "numerical input roles must be unique and completely specified".into(),
            ));
        }
    }
    let semantic = SemanticKeyEnvelope {
        schema_version: 1,
        artifact_kind: contract.kind.into(),
        mathematical_semantics_version: format!(
            "{}/value-v{}",
            contract.kind, contract.value_revision
        ),
        resolved_mathematical_parameters: serde_json::json!({ "parameters": parameters, "inputs": inputs }),
        normalization: parameters.normalization(),
        target: None,
        subspace: parameters.subspace(),
        source_data_identities: BTreeMap::new(),
        algorithm_semantics: None,
    };
    semantic.validate()?;
    Ok(semantic)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_registered_kind_has_one_contract_and_reserved_routes_cannot_produce() {
        let mut seen = BTreeSet::new();
        let mut reserved = 0;
        for family in MANAGED_FAMILIES {
            for kind in crate::artifact_kinds_for_family(family).unwrap() {
                let contract = artifact_identity_contract(kind).unwrap();
                assert_eq!(contract.family, family);
                assert_ne!(contract.payload_codec, "typed-canonical-json");
                if contract.producer.is_none() {
                    assert!(matches!(
                        contract.payload_codec,
                        "reserved" | "registration-only"
                    ));
                }
                assert!(seen.insert(contract.kind));
                reserved += usize::from(contract.reserved);
                crate::artifact_compatibility_policy(family, kind).unwrap();
            }
        }
        assert_eq!(seen.len(), 95);
        assert_eq!(reserved, 6);
        assert!(artifact_identity_contract("unregistered").is_err());
    }

    #[derive(Serialize)]
    struct Parameters {
        precision_bits: u32,
    }
    impl NumericalRequest for Parameters {
        fn artifact_kind(&self) -> &'static str {
            "ccm_tau_matrix"
        }
        fn validate_numerical_parameters(&self) -> Result<(), CacheError> {
            Ok(())
        }
    }
    #[test]
    fn numerical_roles_order_precision_and_content_are_material() {
        let parameters = Parameters {
            precision_bits: 128,
        };
        let input = |role: &str, value: &[u8]| NumericalInputIdentity {
            role: role.into(),
            kind: "ccm_prime_component".into(),
            meaning: "exact-component-coefficients-v1".into(),
            content_digest: ContentDigest::sha256(value),
        };
        let inputs = [input("left", b"a"), input("right", b"b")];
        let digest = numerical_request_key(&parameters, &inputs)
            .unwrap()
            .digest()
            .unwrap();
        let reversed = [inputs[1].clone(), inputs[0].clone()];
        assert_ne!(
            digest,
            numerical_request_key(&parameters, &reversed)
                .unwrap()
                .digest()
                .unwrap()
        );
        assert_ne!(
            digest,
            numerical_request_key(
                &Parameters {
                    precision_bits: 256
                },
                &inputs
            )
            .unwrap()
            .digest()
            .unwrap()
        );
        let mut changed = inputs.clone();
        changed[0].content_digest = ContentDigest::sha256(b"c");
        assert_ne!(
            digest,
            numerical_request_key(&parameters, &changed)
                .unwrap()
                .digest()
                .unwrap()
        );
        assert!(
            numerical_request_key(&parameters, &[inputs[0].clone(), inputs[0].clone()]).is_err()
        );
    }
}
