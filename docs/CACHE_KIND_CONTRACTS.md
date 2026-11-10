# Managed cache kind contracts

The exhaustive source catalog is `xc-cache/src/identity_contract.rs`. Routing alone
does not activate a producer. Active producers retain their existing compliant
semantic builders or use an explicit per-kind value revision. Shared logical
manifest, privacy and compatibility checks apply to every family.

`Registration only` identifies a legacy template or reserved name without an
active managed numerical producer. The six Maynard-Tao kinds have no numerical
implementation in this release. These entries are not claims of runtime execution.

| Family | Kind | Payload codec | Producer status |
| --- | --- | --- | --- |
| quadrature | `gauss_legendre_rule` | `gauss-legendre-rule-v1` | `xc-numerics/quadrature` |
| quadrature | `quadrature_rule` | `quadrature-rule-v1` | `xc-numerics/quadrature` |
| quadrature | `quadrature_reference_table` | `registration-only` | Registration only |
| quadrature | `quadrature_validation` | `registration-only` | Registration only |
| ccm-components | `ccm_prime_enumeration` | `registration-only` | Registration only |
| ccm-components | `ccm_archimedean_integrals` | `portable-archimedean-integrals` | `xc-spectral/ccm/hp` |
| ccm-components | `ccm_archimedean_component` | `registration-only` | Registration only |
| ccm-components | `ccm_prime_component` | `portable-prime-component` | `xc-spectral/ccm/hp` |
| ccm-components | `ccm_pole_component` | `registration-only` | Registration only |
| ccm-matrices | `ccm_tau_matrix` | `portable-tau-matrix-v2` | `xc-spectral/ccm/hp` |
| ccm-matrices | `ccm_even_sector_matrix` | `portable-sector-matrix` | `xc-spectral/ccm/hp` |
| ccm-matrices | `ccm_odd_sector_matrix` | `portable-sector-matrix` | `xc-spectral/ccm/hp` |
| ccm-matrices | `ccm_sector_tridiagonal` | `portable-tridiagonal` | `xc-spectral/ccm/hp` |
| ccm-matrices | `ccm_sector_transform` | `portable-sector-transform` | `xc-spectral/ccm/hp` |
| ccm-matrices | `ccm_reduced_operator` | `registration-only` | Registration only |
| ccm-matrices | `ccm_factorization` | `portable-factorization` | `xc-spectral/ccm/hp` |
| weil-states | `ccm_weil_eigenpair` | `portable-weil-eigenpair` | `xc-spectral/ccm/hp` |
| weil-states | `ccm_sector_eigenvalues` | `portable-sector-eigenvalues` | `xc-spectral/ccm/hp` |
| weil-states | `ccm_sector_spectrum` | `portable-sector-spectrum` | `xc-spectral/ccm/hp` |
| weil-states | `ccm_weil_plunge_state` | `registration-only` | Registration only |
| weil-states | `ccm_weil_sonin_state` | `registration-only` | Registration only |
| weil-states | `ccm_source_eigenbasis` | `registration-only` | Registration only |
| prolate | `prolate_eigenvalue_spectrum` | `portable-prolate-spectrum` | `xc-spectral/prolate` |
| prolate | `prolate_finite_dirichlet_modes` | `portable-dirichlet-modes` | `xc-spectral/prolate` |
| prolate | `prolate_finite_dirichlet_profile` | `portable-dirichlet-profile` | `xc-spectral/prolate` |
| prolate | `ccm_prolate_spectrum` | `registration-only` | Registration only |
| prolate | `ccm_prolate_basis` | `registration-only` | Registration only |
| prolate | `ccm_prolate_candidate` | `registration-only` | Registration only |
| prolate | `ccm_band_concentration` | `registration-only` | Registration only |
| prolate | `ccm_reference_source` | `retained-research-v2` | `xc-spectral/ccm/retained_evidence` |
| prolate | `ccm_external_research_source` | `retained-research-v2` | `xc-spectral/ccm/retained_evidence` |
| ccm-roots | `ccm_secular_source` | `portable-secular-source-v2` | `xc-spectral/ccm/hp` |
| ccm-roots | `ccm_root_count_window` | `registration-only` | Registration only |
| ccm-roots | `ccm_root_discovery_window` | `portable-root-range-v5-v6` | `xc-spectral/ccm/hp` |
| ccm-roots | `ccm_root_refinement` | `portable-root-range-v7-v8` | `xc-spectral/ccm/hp` |
| ccm-roots | `ccm_spectral_window` | `registration-only` | Registration only |
| ccm-evidence | `ccm_compactness_analysis` | `retained-research-v2` | `xc-spectral/ccm/retained_evidence` |
| ccm-evidence | `ccm_arithmetic_energy_analysis` | `retained-research-v2` | `xc-spectral/ccm/retained_evidence` |
| ccm-evidence | `ccm_directional_response_analysis` | `retained-research-v2` | `xc-spectral/ccm/retained_evidence` |
| ccm-evidence | `ccm_weighted_tail_analysis` | `retained-research-v2` | `xc-spectral/ccm/retained_evidence` |
| ccm-evidence | `ccm_spectral_cluster_analysis` | `retained-research-v2` | `xc-spectral/ccm/retained_evidence` |
| ccm-evidence | `ccm_resolution_budget_analysis` | `retained-research-v2` | `xc-spectral/ccm/retained_evidence` |
| ccm-evidence | `ccm_energy_allowance_analysis` | `retained-research-v2` | `xc-spectral/ccm/retained_evidence` |
| ccm-evidence | `ccm_complex_transform_analysis` | `retained-research-v2` | `xc-spectral/ccm/retained_evidence` |
| ccm-evidence | `ccm_root_transport_analysis` | `retained-research-v2` | `xc-spectral/ccm/retained_evidence` |
| ccm-evidence | `ccm_operator_cluster_analysis` | `retained-research-v2` | `xc-spectral/ccm/retained_evidence` |
| ccm-evidence | `ccm_finite_section_transfer` | `retained-research-v2` | `xc-spectral/ccm/retained_evidence` |
| ccm-evidence | `ccm_tail_operator_analysis` | `retained-research-v2` | `xc-spectral/ccm/retained_evidence` |
| ccm-evidence | `ccm_observable_budget_analysis` | `retained-research-v2` | `xc-spectral/ccm/retained_evidence` |
| ccm-evidence | `ccm_capture_preflight` | `retained-research-v2` | `xc-spectral/ccm/retained_evidence` |
| ccm-evidence | `ccm_consistency_analysis` | `retained-research-v2` | `xc-spectral/ccm/retained_evidence` |
| ccm-evidence | `ccm_configuration_comparison` | `retained-research-v2` | `xc-spectral/ccm/retained_evidence` |
| ccm-evidence | `ccm_band_reconstruction` | `retained-research-v2` | `xc-spectral/ccm/retained_evidence` |
| ccm-evidence | `ccm_finite_diagnostic_analysis` | `retained-research-v2` | `xc-spectral/ccm/retained_evidence` |
| ccm-evidence | `ccm_transform_enclosure` | `retained-research-v2` | `xc-spectral/ccm/retained_evidence` |
| ccm-evidence | `research_capture_receipt` | `research-capture-receipt-v1` | `xc-cache/research_artifacts` |
| ccm-evidence | `research_hypothesis_evaluation` | `hypothesis-evaluation-v1` | `xc-cache/research_artifacts` |
| ccm-evidence | `ccm_prefix_analysis` | `prefix-analysis-v1` | `xc-spectral/ccm` |
| ccm-evidence | `ccm_retained_reduction_check` | `retained-reduction-check-v1` | `xc-spectral/ccm` |
| ccm-evidence | `ccm_state_geometry_analysis` | `state-geometry-analysis-v1` | `xc-spectral/ccm` |
| ccm-evidence | `research_observation_packet` | `retained-research-v2` | `xc-spectral/ccm/retained_evidence` |
| ccm-evidence | `research_reference_dataset` | `retained-research-v2` | `xc-spectral/ccm/retained_evidence` |
| ccm-evidence | `ccm_indexed_transform_analysis` | `retained-research-v2` | `xc-spectral/ccm/retained_evidence` |
| ccm-evidence | `ccm_operator_energy_analysis` | `retained-research-v2` | `xc-spectral/ccm/retained_evidence` |
| ccm-evidence | `ccm_root_band_analysis` | `retained-research-v2` | `xc-spectral/ccm/retained_evidence` |
| ccm-evidence | `ccm_stabilization_analysis` | `retained-research-v2` | `xc-spectral/ccm/retained_evidence` |
| ccm-evidence | `ccm_convergence_diagnostics` | `portable-convergence-diagnostics` | `xc-spectral/ccm` |
| ccm-evidence | `ccm_root_conditioning_analysis` | `portable-root-conditioning` | `xc-spectral/ccm` |
| ccm-evidence | `ccm_prime_power_response_analysis` | `portable-prime-response` | `xc-spectral/ccm` |
| ccm-evidence | `ccm_u_flow_response_analysis` | `portable-u-flow-response` | `xc-spectral/ccm` |
| ccm-evidence | `ccm_sector_gap` | `portable-sector-gap` | `xc-spectral/ccm` |
| ccm-evidence | `ccm_sector_gap_certificate` | `dense-or-compact-sector-gap-certificate` | `xc-spectral/ccm` |
| ccm-evidence | `ccm_post_discovery_comparison` | `registration-only` | Registration only |
| ccm-evidence | `ccm_cross_check_record` | `registration-only` | Registration only |
| ccm-evidence | `ccm_validation_record` | `validation-record` | `xc-spectral/ccm` |
| ccm-evidence | `ccm_certificate_bundle` | `portable-certificate-bundle` | `xc-spectral/ccm` |
| ccm-evidence | `ccm_root_certification_report` | `portable-root-certification-report` | `xc-spectral/ccm` |
| ccm-evidence | `ccm_assembly_error_analysis` | `assembly-error-analysis` | `xc-spectral/ccm` |
| ccm-evidence | `ccm_checkpoint_spectra` | `checkpoint-spectra` | `xc-spectral/ccm` |
| ccm-distance | `ccm_weighted_reference_projection` | `retained-research-v2` | `xc-spectral/ccm/retained_evidence` |
| ccm-distance | `ccm_signed_transform_analysis` | `retained-research-v2` | `xc-spectral/ccm/retained_evidence` |
| ccm-distance | `ccm_reference_projection_analysis` | `retained-research-v2` | `xc-spectral/ccm/retained_evidence` |
| ccm-distance | `ccm_deviation_decomposition` | `portable-deviation-decomposition` | `xc-spectral/distance` |
| ccm-distance | `ccm_discretization_distance` | `portable-discretization-distance` | `xc-spectral/distance` |
| ccm-distance | `ccm_distance_resolution_evidence` | `portable-distance-resolution` | `xc-spectral/distance` |
| ccm-distance | `ccm_eigenfunction_profile` | `portable-eigenfunction-profile` | `xc-spectral/distance` |
| ccm-distance | `ccm_target_distance` | `portable-target-distance` | `xc-spectral/distance` |
| ccm-distance | `ccm_target_residual_analysis` | `portable-target-residual` | `xc-spectral/distance` |
| ccm-distance | `ccm_target_comparison_analysis` | `portable-target-comparison` | `xc-spectral/distance` |
| maynard-tao | `maynard_basis` | `reserved` | Reserved |
| maynard-tao | `maynard_moment_table` | `reserved` | Reserved |
| maynard-tao | `maynard_operator` | `reserved` | Reserved |
| maynard-tao | `maynard_candidate` | `reserved` | Reserved |
| maynard-tao | `maynard_bound` | `reserved` | Reserved |
| maynard-tao | `maynard_certificate` | `reserved` | Reserved |

Compatibility floors remain in `compatibility.rs`. New managed manifests use
schema 2; legacy schema-1 bytes remain frozen. A codec name identifies its typed
decoder and may admit multiple explicitly versioned payload layouts. Per-kind
revisions change only when value/evidence meaning changes, never for a release
number or storage destination alone.
