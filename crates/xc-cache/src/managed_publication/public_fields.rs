//! Reviewed field names in the public numerical request vocabulary.
//! A new public producer field must also be added here before publication.
use super::*;

const PARAMETER_NAMES: &str = "\
absolute_tolerance acceptance_rounding accuracy_guard_bits active_prime_power_count
algorithm_semantics allow_incomplete allowance_arithmetic allowance_interpretation allowance_output
alpha analysis_guard_bits arb_available arb_enabled assembly_semantics atom_arithmetic
atom_coordinate_serialization attribution automatic_matrix_admission automatic_matrix_precision
automatic_recipe base_intervals borrowed_dense_action boundary_cluster_resolution_policy
ccm_source_boundary_count certificate_failure certification_scope checkpoint_dimensions
cluster_arithmetic cluster_basis_arithmetic cluster_operator_arithmetic cluster_operator_output
cluster_precision_policy coefficient_count cohort_dependency compactness_arithmetic
comparison_arithmetic comparison_output complement_guard_bits complement_solve completeness
complex_arithmetic complex_fourier_semantics complex_output complex_point_construction
complex_root_point_precision component_selection components consistency_arithmetic
consistency_output consumed_input_contract content_digest convention convergence_confirmation
convergence_defaults coordinate count critical_line_oversampling curvature_output dataset
declared_error_semantics definition degree dense_lu_arithmetic derivative_convention diagnostic
diagnostics dimension dimensions directional_arithmetic directional_output discovery_method
discovery_mode discretization duplicate_coordinate_policy early_termination eigenpair_content_digest
eigenpair_semantic_digest eigenstate_route eigenvalue_count eigenvalue_route
eigenvector_early_termination eigenvector_iteration_limit embedded_residual enclosure_algorithm
enclosure_decimal_output enclosure_fourier_semantics enclosure_point_precision energy_arithmetic
energy_output even_matrix_content_digest even_sector_eigenvalues_content_digest
even_sector_matrix_content_digest even_sector_transform_content_digest even_spectrum_content_digest
event_count exact_contraction_admission expected_rows exponential_rates export_relative_tolerance
export_significant_digits external_input_digest finite_precision_policy finite_transfer_arithmetic
finite_transfer_output first_positive_root_index first_root_index fixed_guard_correction_guards
force_even forced_eigenpair formula full_inertia_semantics full_matrix_inertia_semantics
generation_semantics generator generic_discovery_semantics geometric_terms grid_points grid_variable
ground_index_admission ground_index_validation guards guards_bits half_points hi
higher_eigenpair_content_digest include_primes indexed_prolate_convention innovation_cancellation
input_digest input_preparation_failure inputs integer_cutoff_c interval_matrix_semantics
interval_newton intervals inverse_iteration_semantics inverse_iteration_solver
inverse_iteration_step_limit isolation_bits issue_order kind krylov_guard_eigenpairs
krylov_maximum_restarts krylov_precision_policy krylov_subspace_dimension
krylov_working_precision_bits l2_normalization_arithmetic ladder_positivity_policy lambda
lambda_precision_bits lambda_squared last_root_index left lo logarithm
lower_eigenpair_content_digest m_modes matrices matrix_available matrix_content_digest matrix_policy
maximum_additional_guard_bits maximum_allowance_guard_bits maximum_atom_exponent_span_bits
maximum_atom_guard_bits maximum_bracket_expansions maximum_cluster_guard_bits
maximum_cluster_operator_guard_bits maximum_comparison_guard_bits maximum_complex_guard_bits
maximum_consistency_guard_bits maximum_dimension maximum_directional_guard_bits
maximum_directional_rows maximum_energy_guard_bits maximum_extra_precision_bits
maximum_finite_transfer_guard_bits maximum_iterations maximum_polynomial_band_exact_bits
maximum_signed_band_exact_bits maximum_signed_band_inverse_guard_bits maximum_steps
maximum_tail_form_exact_bits maximum_transform_guard_bits maximum_transport_guard_bits
maximum_weighted_profile_guard_bits meaning members metric model_linear_algebra_arithmetic n n_modes
n_pts natural_eigenpair nodes normalization observation_arithmetic observation_output
odd_spectrum_content_digest options order ordered_basis ordered_sources ordinal pairs parameters
parity parity_basis_arithmetic parity_invariance_premise payload_digest pivot_margin_bits pivoting
point_refinement_semantics points pole_geometry policy policy_digests
polishing_inverse_iteration_step_limit polishing_residual_policy polynomial_band_arithmetic
polynomial_root_output polynomial_root_window positive_root_bound positive_root_scope precision
precision_bits precision_escalation_rule preflight_inventory prime_cutoff profile_recipe
profile_steps projection_arithmetic projection_output quadrature_base_points quadrature_generator
quadrature_points quadrature_policy quadrature_rule recovery_semantics reference
reference_seeds_used relative_change_arithmetic relative_enclosure_bits relative_tolerance report
reported_output_bytes reported_working_bytes representation request requested_eigenpairs
requested_eigenvalues requested_ordinals requested_root_count residual_arithmetic residual_semantics
resolution resolution_arithmetic resolution_output resolved_quadrature_orders resource_admission
response_definition retained_parent_dependencies revision right role root root_count root_domain
root_ordinals root_point_precision root_precision_policy root_range_content_digest
root_selection_digest root_window_policy roots roots_available route rule rule_family rules
run_certificate_available run_sector_certificate run_sector_certificate_representation sample_count
samples scalar_backend scalar_policy schema_version search_depth sector_certificate_representation
sector_eigenvalues_content_digest sector_gap_content_digest sector_matrix_content_digest
sector_transform_content_digest sector_tridiagonal_content_digest secular_source_content_digest
seed_value_identity selected_indices selected_inertia_guard_bits selected_inertia_semantics
selected_root_ordinals semantics sequence signed_band_arithmetic signed_band_inverse_arithmetic
signed_channel_arithmetic solver solver_steps source_accuracy_scope source_boundary_count_semantics
source_dependencies source_dimension source_identity_arithmetic source_inputs source_kind
source_matrix_digest source_parents_are_public source_precision_bits source_status
source_uncertainty_arithmetic source_unit_arithmetic source_weights_digest sources
spectral_isolation_method spectrum_algorithm spectrum_semantics starting_points state_identity
state_normalization_arithmetic stored_resolution_arithmetic streaming_defaults subspace
tail_model_arithmetic tail_model_checkpoint_arithmetic tail_model_householder_arithmetic
tail_model_qr_arithmetic tail_model_vector_recovery_arithmetic target target_precision_bits
tau_content_digest tau_residual_arithmetic tau_source_identity third_inverse_moment
transform_arithmetic transport_arithmetic transport_output tridiagonal_content_digest
tridiagonal_eigenvector_semantics unavailable_cluster_assessment uniform_grid_arithmetic value
value_revision values_digest velocity_parameter verification_precision_bits
weighted_profile_arithmetic weighted_profile_output weights width width_tolerance witness_box_policy
working_precision_bits";

// Targets in public semantic envelopes name numerical operations. Custom
// producers may omit the label or use a reviewed operation.
fn public_target_label(label: &str) -> bool {
    matches!(
        label,
        "actual_retained_state_geometry"
            | "algebraic_minimum"
            | "ccm_configuration_run_summary"
            | "ccm_deviation_decomposition"
            | "ccm_discretization_distance"
            | "ccm_secular_function"
            | "ccm_target_comparison_analysis"
            | "ccm_target_distance"
            | "ccm_target_distance_resolution"
            | "ccm_target_residual_analysis"
            | "complete_parity_sector_eigenvalue_spectrum"
            | "even_sector_weil_form"
            | "odd_sector_weil_form"
            | "finite_ccm_even_odd_sector_gap"
            | "finite_ccm_conditional_parity_simplicity_and_unconditional_full_inertia"
            | "finite_ccm_even_selected_eigenfunction"
            | "finite_research_evidence"
            | "independently_indexed_positive_ccm_roots"
            | "inverse_iteration_linear_solve"
            | "localized_archimedean_form_primitives"
            | "localized_prime_power_component"
            | "localized_weil_form"
            | "lowest_eigenvalues_of_leading_even_blocks"
            | "lowest_parity_sector_eigenpairs"
            | "natural_evenness"
            | "parent_derived_prefix_moments_and_checkpoint_exports"
            | "per_ordinal_finite_source_root_certification"
            | "positive_ccm_spectral_roots"
            | "prolate_wave_operator"
            | "requested_parity_sector_eigenvalue_prefix"
            | "selected_ccm_root_conditioning"
            | "selected_ccm_state_and_root_complete_u_flow"
            | "selected_ccm_state_and_root_prime_velocity_response"
            | "signed_ccm_spectral_roots"
            | "smallest_weil_form_eigenpair"
            | "stored_minus_exact_finite_weil_form"
            | "stored_point_matrix_reduction_diagnostics"
            | "symmetric_tridiagonal_reduction"
            | "tridiagonal_to_dense_eigenvector_transform"
    )
}

fn public_identifier(text: &str) -> bool {
    (1..=64).contains(&text.len())
        && text
            .bytes()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'_')
}

fn digits(text: &str) -> bool {
    !text.is_empty() && text.bytes().all(|c| c.is_ascii_digit())
}

fn doi(id: &str) -> bool {
    let Some((registrant, suffix)) = id.strip_prefix("10.").and_then(|s| s.split_once('/')) else {
        return false;
    };
    (4..=9).contains(&registrant.len())
        && digits(registrant)
        && suffix.split('/').all(|part| {
            !part.is_empty()
                && part != "."
                && part != ".."
                && part
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"-._;():".contains(&c))
        })
}

fn arxiv(id: &str) -> bool {
    let (archive, serial) = id.split_once('/').map_or((None, id), |(a, s)| (Some(a), s));
    let serial = match serial.split_once('v') {
        Some((serial, version)) if digits(version) && !version.starts_with('0') => serial,
        Some(_) => return false,
        None => serial,
    };
    let (date, number) = if let Some(archive) = archive {
        // Legacy arXiv identifiers have a registered archive and YYMMNNN.
        if !matches!(
            archive,
            "acc-phys"
                | "adap-org"
                | "alg-geom"
                | "ao-sci"
                | "astro-ph"
                | "atom-ph"
                | "bayes-an"
                | "chao-dyn"
                | "chem-ph"
                | "cmp-lg"
                | "comp-gas"
                | "cond-mat"
                | "cs"
                | "dg-ga"
                | "funct-an"
                | "gr-qc"
                | "hep-ex"
                | "hep-lat"
                | "hep-ph"
                | "hep-th"
                | "math"
                | "math-ph"
                | "mtrl-th"
                | "nlin"
                | "nucl-ex"
                | "nucl-th"
                | "patt-sol"
                | "physics"
                | "plasm-ph"
                | "q-alg"
                | "q-bio"
                | "quant-ph"
                | "solv-int"
                | "supr-con"
        ) || serial.len() != 7
            || !digits(serial)
        {
            return false;
        }
        serial.split_at(4)
    } else {
        let Some((date, number)) = serial.split_once('.') else {
            return false;
        };
        if !(4..=5).contains(&number.len()) {
            return false;
        }
        (date, number)
    };
    date.len() == 4 && digits(date) && ("01"..="12").contains(&&date[2..]) && digits(number)
}

fn public_reference(text: &str) -> bool {
    match text.split_once(':') {
        Some((prefix, id)) if prefix.eq_ignore_ascii_case("doi") => doi(id),
        Some((prefix, id)) if prefix.eq_ignore_ascii_case("arxiv") => arxiv(id),
        _ => {
            (text.len() == 64 && text.bytes().all(|c| c.is_ascii_hexdigit()))
                || doi(text)
                || arxiv(text)
        }
    }
}

fn finite_decimal(value: &Value) -> bool {
    value.is_number()
        || value
            .as_str()
            .is_some_and(|s| xc_core::DecimalLiteral::validate_str(s).is_ok())
}

fn public_root_target(value: &Value) -> bool {
    let Some(fields) = value.as_object() else {
        return false;
    };
    match value["target"].as_str() {
        Some("prefix") => fields.len() == 2 && value["count"].is_u64(),
        Some("index_range") => {
            fields.len() == 3 && value["first"].is_u64() && value["last"].is_u64()
        }
        Some("positive_height_window") => {
            fields.len() == 3 && finite_decimal(&value["lower"]) && finite_decimal(&value["upper"])
        }
        _ => false,
    }
}

fn public_dataset(value: &Value) -> bool {
    if let Some(label) = value.as_str() {
        return public_identifier(label);
    }
    // This is the numerical DatasetSpec emitted by the indexed transform.
    // Its producer replaces caller attribution before deriving the request.
    value.as_object().is_some_and(|fields| fields.len() == 6)
        && value["schema_version"].as_u64() == Some(1)
        && matches!(
            value["role"].as_str(),
            Some("evaluation_points" | "known_zeta_ordinates" | "retained_ccm_roots")
        )
        && value["attribution"] == "admitted numerical evaluation points"
        && value["coordinate"] == "mellin_t"
        && value["precision_bits"].is_u64()
        && value["points"].as_array().is_some_and(|rows| {
            rows.iter().all(|row| {
                row.as_object().is_some_and(|fields| fields.len() == 3)
                    && row["ordinal"].is_u64()
                    && row.get("value").is_some_and(|v| {
                        v.is_null()
                            || v.as_str()
                                .is_some_and(|s| xc_core::DecimalLiteral::validate_str(s).is_ok())
                    })
                    && row["source_status"].as_str().is_some_and(public_identifier)
            })
        })
}

fn validate_parameters(value: &Value) -> Result<(), CacheError> {
    match value {
        Value::Object(fields) => {
            for (name, value) in fields {
                static NAMES: std::sync::OnceLock<BTreeSet<&str>> = std::sync::OnceLock::new();
                if !NAMES
                    .get_or_init(|| PARAMETER_NAMES.split_ascii_whitespace().collect())
                    .contains(name.as_str())
                {
                    return Err(CacheError::InvalidManifest(format!(
                        "unreviewed public numerical field {name:?}"
                    )));
                }
                // Supplemental input identities and free-form reports belong
                // to private acquisition. Public definitions are fixed formulas.
                let admitted = match name.as_str() {
                    "external_input_digest"
                    | "input_digest"
                    | "values_digest"
                    | "input_preparation_failure"
                    | "report" => value.is_null(),
                    "certificate_failure" => value.is_null() || value == "certificate_unavailable",
                    "definition" => value.as_str().is_some_and(|v| {
                        matches!(v,
                        "canonical even CCM smallest-magnitude eigenfunction sampled on [1, lambda]"
                        | "integral_1^lambda |f_N_lambda(u) - f_M_lambda(u)| u^(-alpha) du"
                        | "log10(abs(lambda_odd)/abs(lambda_even))")
                    }),
                    "formula" => matches!(value.as_str(), Some(
                        "integral_-L/2^L/2 f(x)*exp(-i*t*x) dx; analytic_sinc_and_derivative"
                        | "integral_-L/2^L/2 f(x)*exp(i*t*x) dx; analytic_sinc_and_derivative"
                        | "xi^T Tau xi / xi^T xi; exact_retained_parent"
                        | "abs(new-old)/abs(new); strict comparison before rounding; final two adjacent eligible steps")),
                    "reference" => value.as_str().is_some_and(public_reference),
                    "dataset" => public_dataset(value),
                    "count" => value.is_u64(),
                    "target" => public_root_target(value) || value.as_str().is_some_and(public_target_label),
                    "value" => value.is_boolean() || value.is_i64() || value.is_u64(),
                    "options" => !value
                        .as_array()
                        .is_some_and(|rows| rows.iter().any(|r| r.get("value").is_some())),
                    _ => true,
                };
                if !admitted
                    || value.as_str().is_some_and(|text| {
                        crate::governance::looks_like_local_path(text)
                            || (matches!(
                                name.as_str(),
                                "semantics" | "role" | "kind" | "meaning" | "revision"
                            ) && text.contains(char::is_whitespace))
                    })
                {
                    return Err(CacheError::InvalidManifest(format!(
                        "unreviewed public value for {name:?}"
                    )));
                }
                // These closed forms already validated every nested member in context.
                if !matches!(name.as_str(), "target" | "dataset") {
                    validate_parameters(value)?;
                }
            }
        }
        Value::Array(items) => {
            for value in items {
                validate_parameters(value)?;
            }
        }
        _ => {}
    }
    Ok(())
}

pub(super) fn validate_public_fields(
    semantic: &crate::SemanticKeyEnvelope,
    attestation: &AttestationEnvelope,
) -> Result<(), CacheError> {
    if !matches!(attestation.source_revision.len(), 40 | 64)
        || !attestation
            .source_revision
            .bytes()
            .all(|b| b.is_ascii_hexdigit())
        || attestation.location.is_some()
        || !attestation
            .actor
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
        || attestation.actor.is_empty()
        || attestation.actor.len() > 64
    {
        return Err(CacheError::InvalidManifest(
            "unreviewed public attestation identity or location".into(),
        ));
    }
    for (name, version) in &attestation.dependency_versions {
        let approved = match name.as_str() {
            "xcelerator-toolkit" => crate::ToolkitVersion::parse(version).is_ok(),
            "xcelerator-toolkit-source-sha256" => ContentDigest(version.clone()).validate(),
            _ => false,
        };
        if !approved {
            return Err(CacheError::InvalidManifest(format!(
                "unreviewed public dependency version {name:?}"
            )));
        }
    }
    validate_public_semantics(semantic)
}

pub(super) fn validate_public_semantics(
    semantic: &crate::SemanticKeyEnvelope,
) -> Result<(), CacheError> {
    if semantic
        .target
        .as_deref()
        .is_some_and(|label| !public_target_label(label))
    {
        return Err(CacheError::InvalidManifest(
            "unreviewed public target label".into(),
        ));
    }
    for label in [
        Some(&semantic.mathematical_semantics_version),
        semantic.normalization.as_ref(),
        semantic.subspace.as_ref(),
        semantic.algorithm_semantics.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        if label.is_empty() || label.len() > 512 || !label.is_ascii()
            || label.chars().any(char::is_control)
            || crate::governance::looks_like_local_path(label)
            || label.chars().any(char::is_whitespace) && label != "x=log(u); L=log(lambda_squared); sum_j xi_j exp(2*pi*i*j*(x/L+1/2)); unit_L2_dx; center_positive_else_largest_coefficient_positive"
        {
            return Err(CacheError::InvalidManifest("unreviewed public semantic label".into()));
        }
    }
    validate_parameters(&semantic.resolved_mathematical_parameters)?;
    if let Some(role) = semantic.source_data_identities.keys().find(|role| {
        !matches!(
            role.as_str(),
            "ccm_even_sector_eigenvalues"
                | "ccm_even_sector_matrix"
                | "ccm_even_sector_spectrum"
                | "ccm_even_sector_transform"
                | "ccm_odd_sector_spectrum"
                | "ccm_root_range"
                | "ccm_sector_gap"
                | "ccm_secular_source"
                | "ccm_tau_matrix"
                | "ccm_weil_eigenpair"
                | "complete_spectrum"
                | "selected_modes"
        )
    }) {
        return Err(CacheError::InvalidManifest(format!(
            "unreviewed public numerical source role {role:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn public_parameter_values_are_checked_before_disclosure() {
        use serde_json::json;

        for reference in [
            "private owner notes on the target",
            "private_owner_notes",
            r"Users\example\notes.txt",
            "doi:10.ABCD/value",
            "doi:10.1234/../private",
            "arXiv:2513.04567",
            "arXiv:2501.04567v2 private notes",
        ] {
            assert!(validate_parameters(&json!({"reference": reference})).is_err());
        }
        for dataset in [
            r"C:\Users\example\target.json",
            r"Users\example\target.json",
            "/Library/Application_Support/target.json",
            r"%HOMEPATH%\target.json",
            r"\\server\share\target.json",
            "//server/share/target.json",
            r"Documents\example\target.json",
            "~user/target.json",
        ] {
            assert!(validate_parameters(&json!({"dataset": dataset})).is_err());
        }
        for count in [
            json!("0.31830988618379067153776752674502872406891929148091"),
            json!(2.5),
            json!(-1),
            json!("private notes"),
            json!({"count": 2}),
        ] {
            assert!(
                validate_parameters(&json!({"target":{"target":"prefix","count":count}})).is_err()
            );
        }
        assert!(!public_target_label("private_owner_note_label"));
        assert!(public_target_label("smallest_weil_form_eigenpair"));
        for reference in [
            "doi:10.48550/arXiv.2310.18423",
            "10.5281/zenodo.1234567",
            "arXiv:2501.04567v2",
            "math/0309136",
            &ContentDigest::sha256(b"source").0,
        ] {
            validate_parameters(&json!({"options":[{"reference":reference}]})).unwrap();
        }
        validate_parameters(
            &json!({"dataset":"zeta_zeros_reference_v1","target":{"target":"prefix","count":25}}),
        )
        .unwrap();
        assert!(validate_parameters(&json!({"formula":"private notes"})).is_err());
        assert!(validate_parameters(&json!({"reference":["10.1234/public"]})).is_err());
    }

    #[test]
    fn public_root_certificates_check_types_within_each_target_shape() {
        use serde_json::json;

        for target in [
            json!({"target":"index_range","first":2,"last":5}),
            json!({"target":"positive_height_window","lower":"1e-1000","upper":"30.000000000000000000001"}),
            json!({"target":"positive_height_window","lower":0,"upper":2.5}),
        ] {
            validate_parameters(&json!({"target":target})).unwrap();
        }
        for target in [
            json!({"target":"index_range","first":"2","last":5}),
            json!({"target":"index_range","first":2,"last":-1}),
            json!({"target":"index_range","first":2,"last":5,"note":"private"}),
            json!({"target":"positive_height_window","lower":"private","upper":"30"}),
            json!({"target":"positive_height_window","lower":"NaN","upper":"inf"}),
        ] {
            assert!(validate_parameters(&json!({"target":target})).is_err());
        }
        assert!(validate_parameters(&json!({"lower":"unexpected field"})).is_err());
    }

    #[test]
    fn public_structured_dataset_accepts_only_the_numerical_shape() {
        use serde_json::json;

        let dataset = json!({"schema_version":1,"role":"evaluation_points",
            "attribution":"admitted numerical evaluation points","coordinate":"mellin_t","precision_bits":128,
            "points":[{"ordinal":1,"source_status":"stored_point","value":"1.234567890123456789e-2000"},
                      {"ordinal":2,"source_status":"failed","value":null}]});
        validate_parameters(&json!({"dataset":dataset})).unwrap();
        for (path, private) in [
            ("/attribution", json!("unpublished notes")),
            ("/points/0/source_status", json!(r"Documents\inputs")),
            ("/points/0/value", json!("secret")),
        ] {
            let mut changed = dataset.clone();
            *changed.pointer_mut(path).unwrap() = private;
            assert!(validate_parameters(&json!({"dataset":changed})).is_err());
        }
        let mut extra = dataset;
        extra["points"][0]["reference"] = json!("private notes");
        assert!(validate_parameters(&json!({"dataset":extra})).is_err());
    }

    #[test]
    fn public_fields_require_review_even_when_nested_in_arrays() {
        for value in [
            serde_json::json!({"target":{"value":"3"}}),
            serde_json::json!({"value":"123456789012345678901234567890"}),
            serde_json::json!({"dataset":"private source notes"}),
        ] {
            assert!(validate_parameters(&value).is_err());
        }
        let parameters = serde_json::json!({"precision_bits":128, "options":[{"root_count":1}]});
        validate_parameters(&parameters).unwrap();
        let mut unreviewed = parameters;
        unreviewed["options"][0]["operator_notes"] = serde_json::json!("unreviewed");
        assert!(validate_parameters(&unreviewed).is_err());
    }
}
