//! Closed typed request union for retained producers. JSON is accepted only at
//! the private builder boundary, then decoded with unknown fields rejected.
use super::*;
use crate::ccm::{convergence_capture::finite_capture as fc, extended_research};

#[derive(Serialize)]
pub(super) struct RetainedRequest {
    #[serde(skip)]
    kind: &'static str,
    request: Parameters,
}
#[derive(Serialize)]
#[serde(untagged)]
enum Parameters {
    Reference(ReferenceSpec),
    Dataset(DatasetSpec),
    RootBand(RootBand),
    Transform(Transform),
    Energy(Energy),
    Projection(Projection),
    Observation(ExternalObservation),
    Stabilization(Stabilization),
    TargetSamples(TargetSamples),
    TargetProjection(TargetProjection),
    Acquired(Acquired),
    Consumed(Consumed),
    Extended(Box<Extended>),
    Profiles(Box<Standalone<fc::ProfileRequest>>),
    Cohort(Standalone<fc::RefinementCohort>),
    ExplicitCohort(ExplicitCohort),
    ProjectionEnergy(Box<EnergyRequest<fc::energy_extensions::ProjectionRequest>>),
    ContinuousEnergy(Box<EnergyRequest<fc::energy_extensions::ContinuousRequest>>),
    Finite(Box<Finite>),
}
macro_rules! parameters {
    ($name:ident { $($field:ident : $ty:ty),* $(,)? }) => {
        #[derive(Serialize, Deserialize)]
        #[serde(deny_unknown_fields)]
        struct $name { $($field: $ty),* }
    }
}
parameters!(PreflightInventory {
    lambda_squared: String,
    n_modes: usize,
    source_precision_bits: u32,
    coefficient_count: usize,
    matrix_available: bool,
    roots_available: bool,
    root_count: usize,
    run_certificate_available: bool
});
parameters!(CohortSource {
    kind: String,
    payload_digest: ContentDigest
});
parameters!(CohortMember {state:CohortSource,roots:Option<CohortSource>});
parameters!(CohortPair {left:usize,right:usize,root_ordinals:Vec<usize>,embedded_residual:bool});
parameters!(ExplicitCohort {
    semantics: String,
    members: Vec<CohortMember>,
    pairs: Vec<CohortPair>,
    matrices: Vec<CohortSource>,
    policy_digests: Vec<ContentDigest>
});
parameters!(RootBand {
    rule: String,
    precision_bits: u32
});
parameters!(Transform {
    working_precision_bits: u32,
    dataset: DatasetSpec,
    formula: String,
    transform_arithmetic: String,
    maximum_transform_guard_bits: u32,
    curvature_output: String
});
parameters!(Energy {
    precision_bits: u32,
    formula: String,
    residual_semantics: String
});
parameters!(Projection { options:ProjectionOptions, reference:ContentDigest, ordered_basis:Vec<ContentDigest>,
    metric:String, projection_arithmetic:String, maximum_additional_guard_bits:u32, projection_output:String });
parameters!(Stabilization { options:StabilizationOptions, acceptance_rounding:String,
    relative_change_arithmetic:String, ordered_sources:Vec<ContentDigest> });
parameters!(TargetSamples {
    target_preparation: String,
    target_value: ContentDigest,
    lambda_squared: String,
    precision_bits: u32,
    intervals: usize
});
parameters!(TargetProjection {
    target_preparation: String,
    samples: ContentDigest,
    precision_bits: u32,
    intervals: usize,
    n_modes: usize
});
parameters!(Acquired {
    input_digest: ContentDigest
});
parameters!(Consumed {
    consumed_input_contract: String,
    values_digest: ContentDigest
});

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DiagnosticOptions {
    working_precision_bits: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    exponential_rates: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    maximum_directional_rows: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    relative_tolerance: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    reported_working_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    reported_output_bytes: Option<u64>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Extended {
    semantics: String,
    diagnostic: String,
    expected_rows: Option<usize>,
    options: DiagnosticOptions,
    external_input_digest: Option<ContentDigest>,
    state_identity: Option<NumericalInputIdentity>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    preflight_inventory: Option<PreflightInventory>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    duplicate_coordinate_policy: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    declared_error_semantics: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    ladder_positivity_policy: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    source_unit_arithmetic: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    resource_admission: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    complement_solve: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    complement_guard_bits: Option<Vec<u32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    exact_contraction_admission: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    weighted_profile_arithmetic: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    maximum_weighted_profile_guard_bits: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    weighted_profile_output: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    directional_arithmetic: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    maximum_directional_guard_bits: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    directional_output: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    root_point_precision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    energy_arithmetic: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    component_selection: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    maximum_energy_guard_bits: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    energy_output: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    signed_channel_arithmetic: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cluster_arithmetic: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    maximum_cluster_guard_bits: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cluster_precision_policy: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    unavailable_cluster_assessment: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    atom_arithmetic: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    atom_coordinate_serialization: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    maximum_atom_guard_bits: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    maximum_atom_exponent_span_bits: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    compactness_arithmetic: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    maximum_additional_guard_bits: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    polynomial_band_arithmetic: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    polynomial_root_output: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    polynomial_root_window: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    maximum_polynomial_band_exact_bits: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    signed_band_arithmetic: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    maximum_signed_band_exact_bits: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    signed_band_inverse_arithmetic: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    maximum_signed_band_inverse_guard_bits: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    arb_available: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cohort_dependency: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    enclosure_fourier_semantics: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    enclosure_algorithm: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    enclosure_point_precision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    enclosure_decimal_output: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tail_model_checkpoint_arithmetic: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tail_model_householder_arithmetic: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tail_model_qr_arithmetic: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tail_model_vector_recovery_arithmetic: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tail_model_arithmetic: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    maximum_tail_form_exact_bits: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    transport_arithmetic: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    transport_output: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    maximum_transport_guard_bits: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    model_linear_algebra_arithmetic: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    l2_normalization_arithmetic: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    transform_arithmetic: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    maximum_transform_guard_bits: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    resolution_arithmetic: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    resolution_output: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    finite_transfer_arithmetic: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    finite_transfer_output: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    maximum_finite_transfer_guard_bits: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    consistency_arithmetic: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    consistency_output: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    maximum_consistency_guard_bits: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    complex_fourier_semantics: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    complex_arithmetic: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    complex_point_construction: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    complex_output: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    maximum_complex_guard_bits: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    complex_root_point_precision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cluster_operator_arithmetic: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cluster_operator_output: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    maximum_cluster_operator_guard_bits: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cluster_basis_arithmetic: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    observation_arithmetic: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    observation_output: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    comparison_arithmetic: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    comparison_output: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    maximum_comparison_guard_bits: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    allowance_interpretation: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    allowance_arithmetic: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    maximum_allowance_guard_bits: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    allowance_output: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    run_sector_certificate: Option<ContentDigest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    run_sector_certificate_representation: Option<String>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Standalone<T> {
    semantics: String,
    diagnostic: String,
    input: T,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    profile_recipe: Option<String>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct EnergyRequest<T> {
    semantics: String,
    diagnostic: String,
    request: T,
}
parameters!(ProlateConvention {
    convention: String,
    logarithm: String
});
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Finite {
    semantics: String,
    diagnostic: String,
    input_digest: Option<ContentDigest>,
    certificate_failure: Option<String>,
    input_preparation_failure: Option<String>,
    arb_enabled: bool,
    matrix_policy: String,
    root_window_policy: String,
    automatic_recipe: String,
    profile_recipe: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    streaming_defaults: Option<xc_solver::trial_energy::streaming::Options>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    convergence_defaults: Option<xc_solver::convergence::Options>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    sector_certificate_representation: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    automatic_matrix_admission: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    automatic_matrix_precision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    indexed_prolate_convention: Option<ProlateConvention>,
}
impl RetainedRequest {
    pub(super) fn new(kind: &'static str, value: &Value) -> Result<Self> {
        let request = match kind {
            "ccm_reference_source" => Parameters::Reference(serde_json::from_value(value.clone())?),
            "research_reference_dataset" => {
                Parameters::Dataset(serde_json::from_value(value.clone())?)
            }
            "ccm_root_band_analysis" => {
                Parameters::RootBand(serde_json::from_value(value.clone())?)
            }
            "ccm_indexed_transform_analysis" => {
                Parameters::Transform(serde_json::from_value(value.clone())?)
            }
            "ccm_operator_energy_analysis" => {
                Parameters::Energy(serde_json::from_value(value.clone())?)
            }
            "ccm_reference_projection_analysis" => {
                Parameters::Projection(serde_json::from_value(value.clone())?)
            }
            "research_observation_packet" => {
                Parameters::Observation(serde_json::from_value(value.clone())?)
            }
            "ccm_stabilization_analysis" => {
                Parameters::Stabilization(serde_json::from_value(value.clone())?)
            }
            "ccm_external_research_source" => {
                if let Some(operation) = value.get("target_preparation").and_then(Value::as_str) {
                    match operation {
                        "uniform-log-samples-v1" => {
                            Parameters::TargetSamples(serde_json::from_value(value.clone())?)
                        }
                        "even-fourier-trapezoid-v1" => {
                            Parameters::TargetProjection(serde_json::from_value(value.clone())?)
                        }
                        _ => bail!("unknown target preparation contract"),
                    }
                } else if value.get("consumed_input_contract").is_some() {
                    let request: Consumed = serde_json::from_value(value.clone())?;
                    anyhow::ensure!(
                        matches!(
                            request.consumed_input_contract.as_str(),
                            "per-diagnostic-values-v1" | "finite-diagnostic-values-v1"
                        ),
                        "unknown consumed-input contract"
                    );
                    Parameters::Consumed(request)
                } else {
                    Parameters::Acquired(serde_json::from_value(value.clone())?)
                }
            }
            "ccm_finite_diagnostic_analysis" => {
                match value.get("diagnostic").and_then(Value::as_str) {
                    Some("target_profiles") => {
                        Parameters::Profiles(serde_json::from_value(value.clone())?)
                    }
                    Some("refinement_cohort" | "retained_refinement_cohort") => {
                        Parameters::Cohort(serde_json::from_value(value.clone())?)
                    }
                    Some("projection_energy") => {
                        Parameters::ProjectionEnergy(serde_json::from_value(value.clone())?)
                    }
                    Some("continuous_energy") => {
                        Parameters::ContinuousEnergy(serde_json::from_value(value.clone())?)
                    }
                    Some(
                        "constrained_l1_fit"
                        | "trial_vector_energy"
                        | "trial_vector_parity"
                        | "directional_error_bound"
                        | "finite_tail_bound"
                        | "spectral_cluster_bound"
                        | "finite_root_budget"
                        | "normalization_error_bound"
                        | "continuous_l1_bound"
                        | "dimension_precision_budget"
                        | "indexed_prolate_comparison",
                    ) => Parameters::Finite(serde_json::from_value(value.clone())?),
                    None if value.get("semantics").and_then(Value::as_str)
                        == Some(crate::ccm::explicit_cohort::SEMANTICS) =>
                    {
                        let request: ExplicitCohort = serde_json::from_value(value.clone())?;
                        anyhow::ensure!(
                            request.matrices.len() == request.members.len()
                                && request.policy_digests.len() == request.members.len()
                                && request.policy_digests.iter().all(ContentDigest::validate)
                                && request.matrices.iter().all(|m| {
                                    m.kind == "ccm_tau_matrix" && m.payload_digest.validate()
                                }),
                            "cohort request must bind one Tau payload and policy per member"
                        );
                        Parameters::ExplicitCohort(request)
                    }
                    _ => bail!("finite diagnostic has no typed request contract"),
                }
            }
            _ => {
                let request: Extended = serde_json::from_value(value.clone())?;
                anyhow::ensure!(
                    extended_research::artifact_kind(&request.diagnostic) == Some(kind),
                    "diagnostic and kind contract disagree"
                );
                Parameters::Extended(Box::new(request))
            }
        };
        Ok(Self { kind, request })
    }
}
impl NumericalRequest for RetainedRequest {
    fn artifact_kind(&self) -> &'static str {
        self.kind
    }
    fn validate_numerical_parameters(&self) -> std::result::Result<(), CacheError> {
        let contract = artifact_identity_contract(self.kind)?;
        if contract.payload_codec != "retained-research-v2" {
            return Err(CacheError::InvalidManifest(
                "kind has no retained request codec".into(),
            ));
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn execution_configuration_and_unregistered_producers_cannot_be_requests() {
        let energy = json!({"precision_bits":128,"formula":"finite test form","residual_semantics":"test-v1"});
        assert!(RetainedRequest::new("ccm_operator_energy_analysis", &energy).is_ok());
        let mut accidental = energy.clone();
        accidental["toolkit_version"] = json!("0.17.1");
        assert!(RetainedRequest::new("ccm_operator_energy_analysis", &accidental).is_err());
        assert!(RetainedRequest::new("unregistered", &energy).is_err());
        assert!(RetainedRequest::new("ccm_external_research_source", &json!({"consumed_input_contract":"unknown", "values_digest":ContentDigest::sha256(b"x")})).is_err());
    }
    #[test]
    fn every_finite_capture_route_has_a_strict_request_contract() {
        for id in crate::ccm::capture::FINITE_DIAGNOSTICS {
            let value = json!({"semantics":"finite-fixture","diagnostic":id,"input_digest":null,
                "certificate_failure":null,"input_preparation_failure":null,"arb_enabled":cfg!(feature="arb"),
                "matrix_policy":"fixture","root_window_policy":"fixture","automatic_recipe":"fixture","profile_recipe":"fixture"});
            assert!(
                RetainedRequest::new("ccm_finite_diagnostic_analysis", &value).is_ok(),
                "{id}"
            );
        }
    }
}
