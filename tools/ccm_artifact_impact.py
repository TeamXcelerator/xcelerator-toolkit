#!/usr/bin/env python3
"""Read-only inventory of locally available CCM artifact shard repositories.

This checks metadata, semantic upgrade costs, and a named defect rule. It is NOT a numerical
re-solve, package-hash audit, or certification of unflagged artifacts. It never
contacts GitHub, deletes objects, changes dispositions, or publishes a report.
Do not commit reports containing private manifest identities to public repos.

Usage: python3 tools/ccm_artifact_impact.py /path/to/shard1 /path/to/shard2 \
    --output /private/audit/impact.json
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

HEX = re.compile(r"^[0-9a-f]{64}$")
KNOWN_BAD_SEMANTICS = {
    "ccm-cutoff-free-sector-gap-certificate-v0.14.1-v1",
    "ccm-cutoff-free-sector-gap-certificate-v0.14.1-v2",
}
CORRECTED_SEMANTICS = {
    "ccm-cutoff-free-sector-gap-certificate-mpfr-selected-v5",
    "ccm-cutoff-free-sector-gap-certificate-v0.15.1-v4",
    "ccm-cutoff-free-sector-gap-certificate-v0.15.0-v3",
    # Recognize corrected development drafts without relabeling their identity.
    "ccm-cutoff-free-sector-gap-certificate-v0.14.4-v3",
}
CURRENT_IDENTITIES = {
    "gauss_legendre_rule": "gauss-legendre-v0.13.0-v1",
    "ccm_archimedean_integrals":"ccm-archimedean-integrals-length-aware-order-v3",
    "ccm_prime_component":"ccm-prime-component-v0.15.2-v2",
    "ccm_root_discovery_window": "ccm-root-discovery-v0.15.2-v2",
    "ccm_root_refinement": "ccm-root-range-v0.15.2-v11",
    "ccm_tau_matrix": "ccm-weil-form-source-bound-length-aware-arch-v4",
    "ccm_even_sector_matrix": "ccm-even-sector-v0.15.1-v2",
    "ccm_odd_sector_matrix": "ccm-odd-sector-v0.15.1-v2",
    "ccm_retained_reduction_check": "ccm-retained-reduction-v0.15.1-v3",
    "ccm_prime_power_response_analysis": "ccm-prime-power-response-exact-event-edge-v3",
    "ccm_u_flow_response_analysis": "ccm-u-flow-response-exact-quadrants-v4",
    "ccm_prefix_analysis": "ccm-retained-even-prefix-moments-checked-exports-v8",
    "ccm_sector_tridiagonal": "ccm-parity-tridiagonal-v0.15.0-v1",
    "ccm_sector_transform": "ccm-parity-householder-basis-v0.15.0-v1",
    "ccm_sector_spectrum": "ccm-parity-sector-spectrum-source-enclosures-v3",
    "ccm_sector_eigenvalues": "ccm-parity-sector-eigenvalues-isolating-v4",
    "ccm_sector_gap": "ccm-even-odd-gap-log-source-resolved-v3",
    "ccm_eigenfunction_profile": "ccm-eigenfunction-profile-checked-grid-and-coefficients-v0.15.1-v4",
    "ccm_target_distance": "ccm-runtime-target-distance-exact-weight-checked-coefficients-v0.15.1-v4",
    "prolate_eigenvalue_spectrum": "prolate-bounded-legendre-even-spectrum-v1",
    "ccm_distance_resolution_evidence": "ccm-runtime-target-resolution-evidence-roundtrip-tail-v7-retained-parents-v3",
    "ccm_target_residual_analysis": "ccm-runtime-target-residual-analysis-roundtrip-exact-weight-v6-retained-parents-v3",
    "ccm_deviation_decomposition": "ccm-runtime-target-decomposition-roundtrip-v7-retained-parents-v3",
    "ccm_certificate_bundle": "ccm-exact-point-source-root-certificate-outward-witness-replay-v3",
    'ccm_root_conditioning_analysis': 'ccm-root-conditioning-v0.15.2-v1',
    'ccm_discretization_distance': 'ccm-discretization-distance-roundtrip-pinned-grid-v6',
    'ccm_sector_gap_certificate': 'ccm-cutoff-free-sector-gap-certificate-mpfr-selected-v5',
    'ccm_weil_eigenpair': 'ccm-smallest-weil-eigenpair-stored-resolution-v4',
    'ccm_validation_record': 'ccm-evenness-stored-source-scope-v2',
    'ccm_factorization': 'ccm-dense-lu-v0.13.0-v1',
    'ccm_secular_source': 'ccm-secular-source-v0.13.0-v1',
    'ccm_convergence_diagnostics': 'ccm-run-evidence-stored-resolution-v4',
    'ccm_state_geometry_analysis': 'ccm-retained-fourier-state-geometry-v4',
    'research_capture_receipt': 'managed-research-record-v1',
    'research_hypothesis_evaluation': 'managed-research-record-v1',
    'ccm_arithmetic_energy_analysis': 'ccm-retained-research-observations-v22',
    'ccm_band_reconstruction': 'ccm-retained-research-observations-v22',
    'ccm_capture_preflight': 'ccm-retained-research-observations-v22',
    'ccm_compactness_analysis': 'ccm-retained-research-observations-v22',
    'ccm_complex_transform_analysis': 'ccm-retained-research-observations-v22',
    'ccm_configuration_comparison': 'ccm-retained-research-observations-v22',
    'ccm_consistency_analysis': 'ccm-retained-research-observations-v22',
    'ccm_directional_response_analysis': 'ccm-retained-research-observations-v22',
    'ccm_energy_allowance_analysis': 'ccm-retained-research-observations-v22',
    'ccm_external_research_source': 'ccm-retained-research-observations-v22',
    'ccm_finite_section_transfer': 'ccm-retained-research-observations-v22',
    'ccm_indexed_transform_analysis': 'ccm-retained-research-observations-v22',
    'ccm_observable_budget_analysis': 'ccm-retained-research-observations-v22',
    'ccm_operator_cluster_analysis': 'ccm-retained-research-observations-v22',
    'ccm_operator_energy_analysis': 'ccm-retained-research-observations-v22',
    'ccm_reference_projection_analysis': 'ccm-retained-research-observations-v22',
    'ccm_reference_source': 'ccm-retained-research-observations-v22',
    'ccm_resolution_budget_analysis': 'ccm-retained-research-observations-v22',
    'ccm_root_band_analysis': 'ccm-retained-research-observations-v22',
    'ccm_root_transport_analysis': 'ccm-retained-research-observations-v22',
    'ccm_signed_transform_analysis': 'ccm-retained-research-observations-v22',
    'ccm_spectral_cluster_analysis': 'ccm-retained-research-observations-v22',
    'ccm_stabilization_analysis': 'ccm-retained-research-observations-v22',
    'ccm_tail_operator_analysis': 'ccm-retained-research-observations-v22',
    'ccm_transform_enclosure': 'ccm-retained-research-observations-v22',
    'ccm_weighted_reference_projection': 'ccm-retained-research-observations-v22',
    'ccm_weighted_tail_analysis': 'ccm-retained-research-observations-v22',
    'research_observation_packet': 'ccm-retained-research-observations-v22',
    'research_reference_dataset': 'ccm-retained-research-observations-v22',
}

# Optional prefix diagnostics have a separate, equally current arithmetic identity.
ALTERNATE_IDENTITIES = {
    # Response inversion permits a repeated higher neighbor while still
    # requiring its selected lowest state to be individually isolated.
    'ccm_sector_eigenvalues': {'ccm-even-response-lowest-isolation-with-neighbor-cluster-v1'},
    # The explicitly requested finite-Dirichlet numerical model remains
    # available with its distinct identity; it is not the bounded-endpoint model.
    'prolate_eigenvalue_spectrum': {'prolate-fd-working-precision-spectrum-v0.15.1-v2'},
    'ccm_prefix_analysis': {'ccm-retained-even-prefix-moments-checked-exports-v10'},
    'ccm_root_discovery_window': {'ccm-root-range-v0.15.2-v15'},
    'ccm_root_refinement': {'ccm-root-range-v0.15.2-v12', 'ccm-root-range-v0.15.2-v11-advanced', 'ccm-root-range-v0.15.2-v15'},
    'ccm_weil_eigenpair': {'ccm-smallest-weil-eigenpair-adaptive-even-resolution-v2', 'ccm-smallest-weil-eigenpair-shift-invert-krylov-guarded-resolution-v6'},
    'ccm_convergence_diagnostics': {'ccm-run-evidence-stored-resolution-advanced-v5'},
}
# Named historical transitions supplement the complete current route sets.
# An unlisted label is never assumed to be another current producer route.
SUPERSEDED_IDENTITIES = {
    ('ccm_weil_eigenpair', 'ccm-smallest-weil-eigenpair-shift-invert-krylov-v1'):
        'ccm-smallest-weil-eigenpair-shift-invert-krylov-guarded-resolution-v6',
    ('ccm_weil_eigenpair', 'ccm-smallest-weil-eigenpair-shift-invert-krylov-polished-v2'):
        'ccm-smallest-weil-eigenpair-shift-invert-krylov-guarded-resolution-v6',
    ('ccm_validation_record', 'ccm-evenness-evidence-v0'):
        'ccm-evenness-stored-source-scope-v2',
}
# Literal request arithmetic markers emitted by the retained research producers.
# Source-derived tests require newly added producer markers to enter this table.
CURRENT_REQUEST_PARAMETERS = {'ccm_arithmetic_energy_analysis': {'component_selection': 'explicit_operators_else_compact_actions_v1',
                                    'diagnostic': 'arithmetic_energy',
                                    'energy_arithmetic': 'stored_points_scaled_quadratic_intervals_deficit_kind_v2',
                                    'energy_output': 'midpoints_with_outward_decimal_enclosures_v1',
                                    'maximum_energy_guard_bits': 4096,
                                    'resource_admission': 'resolved_working_bytes_v1',
                                    'semantics': 'extended-retained-diagnostics-v2',
                                    'source_unit_arithmetic': 'binary_scaled_hypot_checked_range_v2'},
 'ccm_band_reconstruction': {'diagnostic': 'band_reconstruction',
                             'exact_contraction_admission': 'all_block_workspace_bound_v1',
                             'ladder_positivity_policy': 'all_required_recurrence_steps_including_failed_v2',
                             'maximum_polynomial_band_exact_bits': 8000000,
                             'maximum_signed_band_exact_bits': 8000000,
                             'maximum_signed_band_inverse_guard_bits': 4096,
                             'maximum_tail_form_exact_bits': 8000000,
                             'polynomial_band_arithmetic': 'stored_polynomial_exact_newton_inverse_moments_v1',
                             'polynomial_root_output': 'outward_root_bounds_and_safe_midpoints_v1',
                             'polynomial_root_window': 'common_binary_scale_exact_rational_cauchy_v2',
                             'resource_admission': 'resolved_working_bytes_v1',
                             'semantics': 'extended-retained-diagnostics-v4',
                             'signed_band_arithmetic': 'declared_points_normalized_recurrence_exact_contractions_v1',
                             'signed_band_inverse_arithmetic': 'relative_zero_guard_scaled_directed_sums_v1',
                             'source_unit_arithmetic': 'binary_scaled_hypot_checked_range_v2',
                             'tail_model_arithmetic': 'declared_points_exact_dyadic_recipe_forms_v2',
                             'tail_model_checkpoint_arithmetic': 'finite-tail-model-original-matrix-dense-source-recovery-v8',
                             'tail_model_householder_arithmetic': 'householder-scaled-opposite-sign-v1',
                             'tail_model_qr_arithmetic': 'tridiag-qr-working-unit-deflation-exponent-safe-hypot-v3',
                             'tail_model_vector_recovery_arithmetic': 'dense-eigenvector-requested-source-rounding-exact-count-scaling-directed-index-gap-angle-v3'},
 'ccm_capture_preflight': {'diagnostic': 'capture_preflight',
                           'resource_admission': 'resolved_working_bytes_v1',
                           'semantics': 'extended-retained-diagnostics-v2',
                           'source_unit_arithmetic': 'binary_scaled_hypot_checked_range_v2'},
 'ccm_compactness_analysis': {'compactness_arithmetic': 'directed_enclosure_agreed_rounding_or_unresolved_v2',
                              'diagnostic': 'compactness',
                              'maximum_additional_guard_bits': 4096,
                              'resource_admission': 'resolved_working_bytes_v1',
                              'semantics': 'extended-retained-diagnostics-v2',
                              'source_unit_arithmetic': 'binary_scaled_hypot_checked_range_v2'},
 'ccm_complex_transform_analysis': {'complex_arithmetic': 'exact_cutoff_original_points_directed_entire_minus_sinc_v2',
                                    'complex_output': 'midpoints_with_outward_decimal_enclosures_v1',
                                    'complex_point_construction': 'original_probes_exact_affine_contour_minus_fourier_v2',
                                    'complex_fourier_semantics': 'retained_fourier_minus_sign_at_original_coordinates_v1',
                                    'complex_root_point_precision': 'declared_payload_precision_v1',
                                    'diagnostic': 'complex_transform',
                                    'maximum_complex_guard_bits': 4096,
                                    'resource_admission': 'resolved_working_bytes_v1',
                                    'semantics': 'extended-retained-diagnostics-v2',
                                    'source_unit_arithmetic': 'binary_scaled_hypot_checked_range_v2',
                                    'workspace_admission': 'configured_row_block_bound_v1'},
 'ccm_configuration_comparison': {'comparison_arithmetic': 'original_points_scaled_overlap_shifted_residual_v1',
                                  'comparison_output': 'midpoints_with_outward_decimal_enclosures_v1',
                                  'diagnostic': 'configuration_comparison',
                                  'duplicate_coordinate_policy': 'all_ambiguous_members_withheld_v2',
                                  'maximum_comparison_guard_bits': 4096,
                                  'maximum_transform_guard_bits': 4096,
                                  'resource_admission': 'resolved_working_bytes_v1',
                                  'semantics': 'extended-retained-diagnostics-v2',
                                  'source_unit_arithmetic': 'binary_scaled_hypot_checked_range_v2',
                                  'transform_arithmetic': 'exact_cutoff_stored_points_directed_sinc_v1'},
 'ccm_consistency_analysis': {'consistency_arithmetic': 'original_points_scaled_action_difference_v1',
                              'consistency_output': 'midpoints_with_outward_decimal_enclosures_v1',
                              'diagnostic': 'consistency',
                              'maximum_consistency_guard_bits': 4096,
                              'resource_admission': 'resolved_working_bytes_v1',
                              'semantics': 'extended-retained-diagnostics-v2',
                              'source_unit_arithmetic': 'binary_scaled_hypot_checked_range_v2'},
 'ccm_directional_response_analysis': {'diagnostic': 'directional_response',
                                       'directional_arithmetic': 'stored_points_projected_resolvent_displacement_checked_v2',
                                       'directional_output': 'midpoints_with_outward_decimal_enclosures_v1',
                                       'maximum_directional_guard_bits': 4096,
                                       'resource_admission': 'resolved_working_bytes_v1',
                                       'root_point_precision': 'declared_payload_precision_v1',
                                       'semantics': 'extended-retained-diagnostics-v2',
                                       'source_unit_arithmetic': 'binary_scaled_hypot_checked_range_v2'},
 'ccm_energy_allowance_analysis': {'allowance_arithmetic': 'declared_points_separate_binary_scales_intervals_v1',
                                   'allowance_interpretation': 'trial-energy-scale-v1',
                                   'allowance_output': 'midpoints_with_outward_decimal_enclosures_v1',
                                   'diagnostic': 'energy_allowance',
                                   'maximum_allowance_guard_bits': 4096,
                                   'resource_admission': 'resolved_working_bytes_v1',
                                   'semantics': 'extended-retained-diagnostics-v2',
                                   'source_unit_arithmetic': 'binary_scaled_hypot_checked_range_v2'},
 'ccm_finite_section_transfer': {'diagnostic': 'finite_section_transfer',
                                 'finite_transfer_arithmetic': 'original_points_scaled_prefix_shifted_residual_v1',
                                 'finite_transfer_output': 'midpoints_with_outward_decimal_enclosures_v1',
                                 'maximum_finite_transfer_guard_bits': 4096,
                                 'resource_admission': 'resolved_working_bytes_v1',
                                 'semantics': 'extended-retained-diagnostics-v2',
                                 'source_unit_arithmetic': 'binary_scaled_hypot_checked_range_v2'},
 'ccm_observable_budget_analysis': {'declared_error_semantics': 'exact_decimal_upper_bound_v2',
                                    'diagnostic': 'observable_budget',
                                    'maximum_transform_guard_bits': 4096,
                                    'observation_arithmetic': 'original_points_directed_l2_transport_v1',
                                    'observation_output': 'allowance_upper_margin_lower_with_expression_enclosures_v1',
                                    'resource_admission': 'resolved_working_bytes_v1',
                                    'semantics': 'extended-retained-diagnostics-v2',
                                    'source_unit_arithmetic': 'binary_scaled_hypot_checked_range_v2',
                                    'transform_arithmetic': 'exact_cutoff_stored_points_directed_sinc_v1'},
 'ccm_operator_cluster_analysis': {'cluster_basis_arithmetic': 'original_points_unit_columns_before_rank_threshold_v1',
                                   'cluster_operator_arithmetic': 'original_points_shift_before_interval_projection_lu_v1',
                                   'cluster_operator_output': 'midpoints_with_outward_decimal_enclosures_v1',
                                   'diagnostic': 'operator_cluster',
                                   'maximum_cluster_operator_guard_bits': 4096,
                                   'resource_admission': 'resolved_working_bytes_v1',
                                   'semantics': 'extended-retained-diagnostics-v2',
                                   'source_unit_arithmetic': 'binary_scaled_hypot_checked_range_v2'},
 'ccm_operator_energy_analysis': {'residual_semantics': 'raw_matrix_explicit_zero_eigenvalue_normalization_v2'},
 'ccm_resolution_budget_analysis': {'diagnostic': 'resolution_budget',
                                    'maximum_transform_guard_bits': 4096,
                                    'resolution_arithmetic': 'exact_decimal_tolerance_original_point_conditional_distance_v2',
                                    'resolution_output': 'outward_allowance_upper_endpoints_with_expression_enclosures_v1',
                                    'resource_admission': 'resolved_working_bytes_v1',
                                    'semantics': 'extended-retained-diagnostics-v3',
                                    'source_unit_arithmetic': 'binary_scaled_hypot_checked_range_v2',
                                    'transform_arithmetic': 'exact_cutoff_stored_points_directed_sinc_v1'},
 'ccm_root_transport_analysis': {'diagnostic': 'root_transport',
                                 'maximum_transport_guard_bits': 4096,
                                 'resource_admission': 'resolved_working_bytes_v1',
                                 'semantics': 'extended-retained-diagnostics-v2',
                                 'source_unit_arithmetic': 'binary_scaled_hypot_checked_range_v2',
                                 'transport_arithmetic': 'exact_cutoff_stored_points_directional_intervals_v1',
                                 'transport_output': 'midpoints_with_outward_decimal_enclosures_v1'},
 'ccm_signed_transform_analysis': {'diagnostic': 'signed_transform',
                                   'maximum_transform_guard_bits': 4096,
                                   'resource_admission': 'resolved_working_bytes_v1',
                                   'semantics': 'extended-retained-diagnostics-v2',
                                   'signed_channel_arithmetic': 'exact_cutoff_center_declared_points_checked_channels_v3',
                                   'source_unit_arithmetic': 'binary_scaled_hypot_checked_range_v2',
                                   'transform_arithmetic': 'exact_cutoff_stored_points_directed_sinc_v1'},
 'ccm_spectral_cluster_analysis': {'cluster_arithmetic': 'stored_points_scaled_unit_checked_gram_v3',
                                   'cluster_precision_policy': 'unit_column_pivot_proxy_v1',
                                   'diagnostic': 'spectral_cluster',
                                   'maximum_cluster_guard_bits': 4096,
                                   'resource_admission': 'resolved_working_bytes_v1',
                                   'semantics': 'extended-retained-diagnostics-v2',
                                   'source_unit_arithmetic': 'binary_scaled_hypot_checked_range_v2'},
 'ccm_tail_operator_analysis': {'declared_error_semantics': 'exact_decimal_upper_bound_v2',
                                'diagnostic': 'tail_operator',
                                'l2_normalization_arithmetic': 'power_two_scaled_max_precision_l2_v2',
                                'maximum_tail_form_exact_bits': 8000000,
                                'model_linear_algebra_arithmetic': 'exact_stored_dot_product_stages_and_tail_bound_v0.15.2-v2',
                                'resource_admission': 'resolved_working_bytes_v1',
                                'semantics': 'extended-retained-diagnostics-v4',
                                'source_unit_arithmetic': 'binary_scaled_hypot_checked_range_v2',
                                'tail_model_arithmetic': 'declared_points_exact_dyadic_recipe_forms_v2',
                             'tail_model_checkpoint_arithmetic': 'finite-tail-model-original-matrix-dense-source-recovery-v8',
                             'tail_model_householder_arithmetic': 'householder-scaled-opposite-sign-v1',
                             'tail_model_qr_arithmetic': 'tridiag-qr-working-unit-deflation-exponent-safe-hypot-v3',
                             'tail_model_vector_recovery_arithmetic': 'dense-eigenvector-requested-source-rounding-exact-count-scaling-directed-index-gap-angle-v3'},
 'ccm_transform_enclosure': {'diagnostic': 'transform_enclosure',
                             'enclosure_algorithm': 'centered-taylor-48-integral-remainder-minus-fourier-v4-stored-points',
                             'enclosure_fourier_semantics': 'retained_fourier_minus_sign_at_original_coordinates_v1',
                             'enclosure_decimal_output': 'outward_endpoints_v1',
                             'enclosure_point_precision': 'declared_payload_and_external_precision_v1',
                             'resource_admission': 'resolved_working_bytes_v1',
                             'semantics': 'extended-retained-diagnostics-v5-minus-fourier-source-error-hull',
                             'source_unit_arithmetic': 'binary_scaled_hypot_checked_range_v2'},
 'ccm_weighted_reference_projection': {'diagnostic': 'weighted_reference_projection',
                                       'maximum_weighted_profile_guard_bits': 4096,
                                       'resource_admission': 'resolved_working_bytes_v1',
                                       'semantics': 'extended-retained-diagnostics-v2',
                                       'source_unit_arithmetic': 'binary_scaled_hypot_checked_range_v2',
                                       'weighted_profile_arithmetic': 'stored_points_combined_difference_interval_gram_unresolved_v2',
                                       'weighted_profile_output': 'midpoint_with_outward_decimal_enclosures_v1'},
 'ccm_weighted_tail_analysis': {'atom_arithmetic': 'stored_points_exact_mass_directed_moments_v2',
                                'atom_coordinate_serialization': 'promoted_source_point_v1',
                                'diagnostic': 'weighted_tail',
                                'maximum_atom_exponent_span_bits': 1000000,
                                'maximum_atom_guard_bits': 4096,
                                'resource_admission': 'resolved_working_bytes_v1',
                                'semantics': 'extended-retained-diagnostics-v4',
                                'source_unit_arithmetic': 'binary_scaled_hypot_checked_range_v2'}}

CURRENT_PARAMETERS = {'ccm_sector_spectrum': {
    'tridiagonal_eigenvector_semantics': 'tridiag-interleaved-requested-source-rounding-exact-count-scaling-directed-index-gap-angle-v10',
    'eigenvector_early_termination': True,
}, 'ccm_weil_eigenpair': {
    'source_identity_arithmetic': 'validated_tau_dependency_and_exact_log_source_v2',
    'ground_index_validation': 'directed_stored_source_ground_index_and_rounding_scale_gap_v2',
    'tau_residual_arithmetic': 'directed_scaled_stored_infinity_norm_v2',
    'parity_basis_arithmetic': 'directed_stored_point_orthonormal_parity_v2',
    'state_normalization_arithmetic': 'directed_scaled_exact_sum_sqrt_l_state_v2',
    'l2_normalization_arithmetic': 'power_two_scaled_max_precision_l2_v2',
    'inverse_iteration_semantics': 'dense-inverse-iteration-mixed-parity-residual-checked-wrapper-v3',
}}


GRID_KINDS = {
    "ccm_eigenfunction_profile", "ccm_target_distance", "ccm_distance_resolution_evidence",
    "ccm_target_residual_analysis", "ccm_deviation_decomposition", "ccm_discretization_distance",
}
for _kind in {"ccm_root_discovery_window", "ccm_root_refinement"}:
    CURRENT_PARAMETERS[_kind] = {
        "point_refinement_semantics": "root-point-positional-stop-local-nondecreasing-residual-growth-v3",
        "generic_discovery_semantics": "root-discovery-disjoint-bracket-distinctness-v2",
    }
for _kind in GRID_KINDS:
    CURRENT_PARAMETERS[_kind] = {"uniform_grid_arithmetic": "uniform-grid-relative-log-span-pinned-endpoints-v3"}

# Changes to algorithm_semantics are semantic-key changes even if the version
# string is unchanged. Match complete emitted forms, not arbitrary substrings.
QR_SEMANTICS = "tridiag-qr-working-unit-deflation-exponent-safe-hypot-v3"
CURRENT_ALGORITHMS = {
    "prolate_eigenvalue_spectrum": {QR_SEMANTICS},
    "ccm_sector_eigenvalues": {
        "hp_sturm_per_index_adaptive_bisection_binary_scaled_tolerance_v4+" + QR_SEMANTICS,
        QR_SEMANTICS + "+" + QR_SEMANTICS,
        "implicit_wilkinson_shift_tridiagonal_qr_cross_checked_by_per_index_hp_sturm_v5+" + QR_SEMANTICS,
    },
    "ccm_root_conditioning_analysis": {"directed_scaled_correct_rounding_and_complete_uniform_pole_geometry_v3"},
}
FIXED_CORRECTION_GUARDS = [64,128,256,512,1024,2048,4096]
ROOT_KINDS = {"ccm_root_discovery_window", "ccm_root_refinement"}

def additional_identity_mismatches(record: dict[str, Any]) -> list[str]:
    """Version-independent and conditional producer-key changes only.

    This is a release-impact census, not validation of arbitrary numerical
    requests. Dimensions, input data and caller-selected policies remain data.
    """
    kind = record["artifact_kind"]
    p = record["resolved_mathematical_parameters"]
    algorithm = record.get("algorithm_semantics")
    reasons = []
    if kind in CURRENT_ALGORITHMS and algorithm not in CURRENT_ALGORITHMS[kind]:
        reasons.append("algorithm_semantics")
    if kind == 'ccm_sector_spectrum':
        # The limit is caller data; its presence binds the execution option.
        limit = p.get('eigenvector_iteration_limit')
        if type(limit) is not int or limit < 1:
            reasons.append('eigenvector_iteration_limit')
    if kind in ROOT_KINDS:
        suffix = "+adaptive_root_precision_v1" if p.get("root_precision_policy") == "adaptive_v1" else "+confirmed_retained_correction_v2"
        if not isinstance(p.get("solver"), str) or algorithm != p["solver"] + suffix:
            reasons.append("root_algorithm_semantics")
        if p.get("root_precision_policy") != "adaptive_v1" and p.get("fixed_guard_correction_guards") != FIXED_CORRECTION_GUARDS:
            reasons.append("fixed_guard_correction_guards")
    if kind == "ccm_indexed_transform_analysis":
        request = p.get("request", {})
        if not isinstance(request, dict):
            reasons.append("transform_request")
        else:
            formulas = {
                "integral_-L/2^L/2 f(x)*exp(-i*t*x) dx; analytic_sinc_and_derivative": "exact_cutoff_stored_points_directed_sinc_root_convention_v2",
                "integral_-L/2^L/2 f(x)*exp(i*t*x) dx; analytic_sinc_and_derivative": "exact_cutoff_stored_points_directed_sinc_v1",
            }
            expected = formulas.get(request.get("formula"))
            if expected is None or request.get("transform_arithmetic") != expected:
                reasons.append("transform_arithmetic")
    return reasons


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def inventory(shards: list[Path], verify_manifest_bytes: bool = False, registries: list[Path] | None = None) -> dict[str, Any]:
    if not shards:
        raise ValueError("at least one local shard root is required")
    records: dict[str, dict[str, Any]] = {}
    active: set[str] = set()
    metadata_errors: list[str] = []
    indexes_seen = 0
    manifests_seen = 0
    roots_seen: set[Path] = set()
    repositories_seen: set[str] = set()
    for original in shards:
        root = original.resolve(strict=True)
        if not root.is_dir() or root in roots_seen:
            raise ValueError(f"not a unique shard directory: {root}")
        roots_seen.add(root)
        config = read_json(root / "cache-repository.json")
        repository = config.get("repository", root.name)
        if not isinstance(repository,str) or not repository.strip():
            raise ValueError(f"shard has no valid repository identity: {root}")
        repositories_seen.add(repository)
        index_paths = sorted((root / "indexes").glob("*/*.json"))
        manifest_paths = sorted((root / "manifests").glob("*/*.json"))
        if not index_paths and manifest_paths:
            metadata_errors.append(f"{repository}: manifests exist but no managed indexes found")
        for path in index_paths:
            indexes_seen += 1
            try:
                data = read_json(path)
                entries = data.get("entries")
                if not isinstance(entries, list):
                    raise ValueError("index has no entries list")
                index_active = set()
                for item in entries:
                    digest = item.get("manifest_digest", "")
                    if not isinstance(digest, str) or not HEX.fullmatch(digest):
                        raise ValueError("invalid index manifest digest")
                    if item.get("disposition") == "active":
                        index_active.add(digest)
                active.update(index_active)
            except (OSError, ValueError, TypeError, AttributeError) as error:
                metadata_errors.append(f"{repository}:{path.relative_to(root)}: {error}")
        for path in manifest_paths:
            manifests_seen += 1
            try:
                digest = path.stem
                if not HEX.fullmatch(digest):
                    raise ValueError("manifest filename is not a SHA-256 identity")
                raw = path.read_bytes()
                data = json.loads(raw)
                if not isinstance(data, dict):
                    raise ValueError("manifest is not a JSON object")
                if verify_manifest_bytes and hashlib.sha256(raw).hexdigest() != digest:
                    raise ValueError("manifest raw-byte SHA-256 differs from filename")
                semantic = data.get("semantic_key")
                payload = data.get("canonical_payload")
                if not isinstance(semantic, dict) or not isinstance(payload, dict):
                    raise ValueError("manifest has no semantic key or canonical payload")
                for field in ('artifact_kind','mathematical_semantics_version'):
                    if not isinstance(semantic.get(field),str) or not semantic[field].strip():
                        raise ValueError(f'invalid semantic {field}')
                if semantic.get('algorithm_semantics') is not None and not isinstance(semantic['algorithm_semantics'], str):
                    raise ValueError('semantic algorithm must be a string or null')
                parameters = semantic.get('resolved_mathematical_parameters',{})
                if not isinstance(parameters,dict):
                    raise ValueError('semantic mathematical parameters must be an object')
                dependencies = payload.get("dependencies")
                if not isinstance(dependencies, list):
                    raise ValueError("manifest dependencies are not a list")
                dependency_digests = []
                for dep in dependencies:
                    value = dep.get("manifest_digest", "")
                    if not isinstance(value, str) or not HEX.fullmatch(value):
                        raise ValueError("invalid dependency manifest digest")
                    dependency_digests.append(value)
                record = {
                    "manifest_digest": digest,
                    "repository": repository,
                    "path": str(path.relative_to(root)),
                    "artifact_family": data.get("artifact_family"),
                    "artifact_kind": semantic.get("artifact_kind"),
                    "mathematical_semantics_version": semantic.get("mathematical_semantics_version"),
                    "producer_toolkit_version": data.get("producer_toolkit_version"),
                    "dependencies": dependency_digests,
                    "resolved_mathematical_parameters": parameters,
                    "algorithm_semantics": semantic.get("algorithm_semantics"),
                    "manifest_byte_hash_checked": verify_manifest_bytes,
                }
                previous = records.get(digest)
                if previous is not None:
                    comparable = {key: value for key, value in record.items() if key not in {"repository", "path"}}
                    old = {key: value for key, value in previous.items() if key not in {"repository", "path"}}
                    if old != comparable:
                        raise ValueError("same manifest identity has inconsistent metadata across shards")
                else:
                    records[digest] = record
            except (OSError, ValueError, TypeError, AttributeError) as error:
                metadata_errors.append(f"{repository}:{path.relative_to(root)}: {error}")

    missing_active = sorted(active - records.keys())
    reverse: dict[str, set[str]] = defaultdict(set)
    missing_dependencies: set[str] = set()
    directly_affected: set[str] = set()
    needs_review: set[str] = set()
    identity_changes: set[str] = set()
    uncovered: set[str] = set()
    for digest, record in records.items():
        for dep in record["dependencies"]:
            reverse[dep].add(digest)
            if dep not in records:
                missing_dependencies.add(dep)
        if record["artifact_kind"] == "ccm_sector_gap_certificate":
            version = record["mathematical_semantics_version"]
            if version in KNOWN_BAD_SEMANTICS:
                directly_affected.add(digest)
            elif version not in CORRECTED_SEMANTICS:
                needs_review.add(digest)
        current = CURRENT_IDENTITIES.get(record["artifact_kind"])
        if current is None:
            uncovered.add(digest)
        allowed = {current} | ALTERNATE_IDENTITIES.get(record['artifact_kind'],set())
        stale_version = (
            current is not None and record['mathematical_semantics_version'] not in allowed
        ) or (record['artifact_kind'], record['mathematical_semantics_version']) in SUPERSEDED_IDENTITIES
        required = CURRENT_PARAMETERS.get(record['artifact_kind'],{})
        stale_parameters = any(record['resolved_mathematical_parameters'].get(key) != value for key,value in required.items())
        request = record['resolved_mathematical_parameters'].get('request', {})
        required_request = CURRENT_REQUEST_PARAMETERS.get(record['artifact_kind'], {})
        stale_request = bool(required_request) and (not isinstance(request, dict) or any(
            request.get(key) != value for key, value in required_request.items()))
        additional = additional_identity_mismatches(record)
        record["upgrade_identity_mismatches"] = (
            (["mathematical_semantics_version"] if stale_version else [])
            + ["parameters." + key for key,value in required.items()
               if record["resolved_mathematical_parameters"].get(key) != value]
            + (["request_parameters"] if stale_request else []) + additional
        )
        if stale_version or stale_parameters or stale_request or additional:
            identity_changes.add(digest)

    affected = set(directly_affected)
    queue = deque(sorted(directly_affected))
    while queue:
        parent = queue.popleft()
        for child in sorted(reverse.get(parent, ())):
            if child not in affected:
                affected.add(child)
                queue.append(child)
    recompute = set(identity_changes)
    queue = deque(sorted(identity_changes))
    while queue:
        for child in sorted(reverse.get(queue.popleft(), ())):
            if child not in recompute:
                recompute.add(child)
                queue.append(child)
    required_repositories: set[str] = set()
    for registry in registries or []:
        # Registry shard descriptors enumerate both archived and current shards.
        registry = registry.resolve(strict=True)
        descriptors = sorted((registry / "shards").glob("*.json"))
        if not descriptors:
            raise ValueError(f"registry has no shard descriptors: {registry}")
        for path in descriptors:
            repository = read_json(path).get("repository")
            if not isinstance(repository, str) or not repository.strip():
                raise ValueError(f"shard descriptor has no repository identity: {path}")
            required_repositories.add(repository)
    missing_repositories = sorted(required_repositories - repositories_seen)
    listed = []
    for digest in sorted(records):
        record = records[digest].copy()
        record["active"] = digest in active
        record["impact"] = (
            "known_defective_assembly_semantics" if digest in directly_affected
            else "depends_on_known_defective_artifact" if digest in affected
            else "unrecognized_certificate_semantics_requires_review" if digest in needs_review
            else "not_flagged_by_this_defect_rule"
        )
        record["upgrade_reuse"] = (
            "new_semantic_identity_required" if digest in identity_changes else
            "new_parent_identity_required" if digest in recompute else
            "unrecognized_artifact_kind_requires_review" if digest in uncovered else
            "not_flagged_by_upgrade_rules"
        )
        listed.append(record)
    return {
        "schema_version": 2,
        "algorithm_semantics": "ccm-impact-current-source-epochs-v3",
        "scope": "read_only_metadata_impact_inventory_not_numerical_validation",
        "defect_rule": "cutoff_free_zero_mode_omitted_finite_endpoint_correction",
        "shards_requested": len(shards),
        "indexes_read": indexes_seen,
        "manifest_files_read": manifests_seen,
        "unique_manifests": len(records),
        "active_manifests": len(active),
        "kind_counts": dict(sorted(Counter(r["artifact_kind"] for r in records.values()).items())),
        "directly_affected_count": len(directly_affected),
        "affected_including_descendants_count": len(affected),
        "active_affected_count": len(active & affected),
        "active_unrecognized_certificate_count": len(active & needs_review),
        "active_recompute_count": len(active & recompute),
        "active_uncovered_kind_count": len(active & uncovered),
        "uncovered_artifact_kinds": sorted({records[d]["artifact_kind"] for d in uncovered}),
        "upgrade_rule_coverage_complete_for_supplied_kinds": not uncovered,
        "active_recompute_by_kind": dict(sorted(Counter(records[d]["artifact_kind"] for d in active & recompute).items())),
        "upgrade_interpretation": "Identity changes require fresh computations when requested, not deletion or a finding of byte corruption. Unknown artifact kinds and missing dependencies can undercount upgrade recomputations; the report explicitly separates metadata coverage and rule coverage. Runtime cost depends on each dimension and precision; artifact counts are not timings.",
        "missing_registry_repositories": missing_repositories,
        "inventory_roots": [str(root) for root in sorted(roots_seen)],
        "missing_active_manifest_digests": missing_active,
        "missing_dependency_manifest_digests": sorted(missing_dependencies),
        "metadata_errors": metadata_errors,
        "metadata_coverage_complete_for_supplied_shards": not metadata_errors and not missing_active and not missing_dependencies and not missing_repositories,
        "coverage_complete_for_supplied_shards": not metadata_errors and not missing_active and not missing_dependencies and not missing_repositories and not uncovered,
        "payload_packages_verified": False,
        "numerical_results_recomputed": False,
        "additional_repair_limits": [
            "A recognized current label is not a full semantic-key or payload validation. The listed arithmetic/algorithm markers cover known producer changes; caller inputs and all runtime admission rules are not reimplemented here.",
            "CCM Sonin basis correction has no dedicated managed result cache; caller-retained outputs require separate provenance review.",
            "Resolution and complete-discovery identity changes flag replay requirements, not proof that every prior numerical result was wrong.",
        ],
        "artifacts": listed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--registry", type=Path, action="append", default=[], help="local registry checkout; enumerate absent rollover shards (repeat for both visibilities)")
    parser.add_argument("--verify-manifest-bytes", action="store_true",
                        help="also hash raw manifest files; does not verify numerical packages")
    args = parser.parse_args()
    try:
        report = inventory(args.shards, args.verify_manifest_bytes, args.registry)
        output = args.output.resolve()
        if any(output == root.resolve() or root.resolve() in output.parents for root in args.shards):
            raise ValueError("write the audit report outside the immutable shard directories")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(output)
    except (OSError, ValueError) as error:
        print(f"inventory failed: {error}", file=sys.stderr)
        return 2
    print(f"Read {report['unique_manifests']} unique manifests across {report['shards_requested']} shards.")
    print(f"Active affected: {report['active_affected_count']}; active certificate semantics needing review: {report['active_unrecognized_certificate_count']}.")
    print("Unflagged does not mean numerically verified.")
    print(f"Active upgrade recomputations: {report['active_recompute_count']}; missing registry repositories: {len(report['missing_registry_repositories'])}.")
    if (report["active_affected_count"] or report["active_unrecognized_certificate_count"]
            or report["active_recompute_count"] or not report["coverage_complete_for_supplied_shards"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
