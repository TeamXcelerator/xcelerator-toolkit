//! Explicit consumed-input views. The complete acquisition bundle is provenance.
//! Each variant is reviewed alongside its evaluator; no JSON field blacklist.
use super::*;
use crate::ccm::{convergence_capture as cc, research_completion as completion};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "diagnostic", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum ExternalValues {
    WeightedReferenceProjection {
        precision: u32,
        intervals: usize,
        values: Vec<String>,
        basis_values: Vec<Vec<String>>,
        fixed_second_component: Option<String>,
        raw_normalizer: String,
        approximation_scope: String,
    },
    SignedTransform {
        precision: u32,
        jets: Vec<SignedJet>,
    },
    ArithmeticEnergy {
        precision: u32,
        operators: Vec<OperatorComponent>,
        actions: Vec<cc::OperatorAction>,
        complete: bool,
        trial: Option<Vec<String>>,
        deficit: Option<String>,
        deficit_kind: Option<String>,
    },
    DirectionalResponse {
        precision: u32,
        perturbations: Vec<OperatorComponent>,
        actions: Vec<cc::OperatorAction>,
    },
    WeightedTail {
        precision: u32,
        atoms: Vec<WeightedAtom>,
        coordinate: Option<String>,
        coverage: Option<String>,
        checkpoints: Vec<String>,
        evaluations: Vec<crate::ccm::atom_research::AtomEvaluation>,
    },
    SpectralCluster {
        precision: u32,
        cluster: Vec<ClusterVector>,
        previous_cluster: Vec<ClusterVector>,
        boundary: Option<[String; 2]>,
        assessment: Option<ClusterPreparationAssessment>,
    },
    ResolutionBudget {
        precision: u32,
        jets: Vec<ResolutionJet>,
    },
    EnergyAllowance {
        precision: u32,
        allowance: EnergyAllowance,
    },
    RootTransport {
        precision: u32,
        perturbations: Vec<OperatorComponent>,
        actions: Vec<cc::OperatorAction>,
        log_cutoff_velocity: Option<String>,
        checks: Vec<completion::ResponseCheck>,
    },
    OperatorCluster {
        precision: u32,
        reference_vectors: Vec<Vec<String>>,
    },
    FiniteSectionTransfer {
        comparison: cc::ComparisonState,
    },
    TailOperator {
        precision: u32,
        form: Option<cc::TailForm>,
        atoms: Vec<WeightedAtom>,
        recipe: Option<completion::TailFormRecipe>,
        cutoffs: Vec<String>,
        definition: ContentDigest,
    },
    ObservableBudget {
        uncertainty: cc::SourceUncertainty,
    },
    CapturePreflight {
        sampled_reference: bool,
        reference_jets: bool,
        atoms: bool,
        independent_actions: bool,
        comparisons: bool,
        band: bool,
        sector_certificate: bool,
        tail_form: bool,
    },
    Consistency {
        precision: u32,
        components: Vec<cc::OperatorAction>,
        independent_actions: Vec<cc::OperatorAction>,
    },
    ConfigurationComparison {
        comparisons: Vec<completion::ComparisonSnapshot>,
    },
    BandReconstruction {
        precision: u32,
        band: Option<completion::SignedBandModel>,
        polynomial_tail: Option<cc::TailForm>,
        cutoffs: Vec<String>,
    },
    TransformEnclosure {
        precision: u32,
        contour: Option<completion::ContourPolicy>,
        certificate:
            Option<Box<crate::ccm::sector_gap_certificate::PortableCcmSectorGapCertificate>>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct SignedJet {
    ordinal: usize,
    t: String,
    window: Jet,
    full: Jet,
    tail: Jet,
    endpoint: Option<Jet>,
    fitted: Vec<Jet>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct ResolutionJet {
    ordinal: usize,
    t: String,
    matched_root_ordinal: Option<usize>,
    error_normalization: Option<String>,
    source_value_error: Option<String>,
    tail_value_error: Option<String>,
    source_derivative_error: Option<String>,
    root_separation_radius: Option<String>,
}

pub(super) fn external_values(
    id: &str,
    input: Option<&ExternalResearchInputs>,
) -> Result<Option<ExternalValues>> {
    use ExternalValues::*;
    if matches!(id, "compactness" | "complex_transform") {
        return Ok(None);
    }
    let Some(i) = input else { return Ok(None) };
    let precision = i.precision_bits;
    let empty_run = cc::RunOnceInputs::default();
    let run = i.run_once.as_ref().unwrap_or(&empty_run);
    let empty_completion = completion::CompletionInputs::default();
    let c = run.completion.as_ref().unwrap_or(&empty_completion);
    let policy = c.atom_analysis.as_ref();
    let cutoffs = || policy.map(|p| p.cutoffs.clone()).unwrap_or_default();
    Ok(Some(match id {
        "weighted_reference_projection" => {
            let Some(t) = &i.target else { return Ok(None) };
            WeightedReferenceProjection {
                precision,
                intervals: t.intervals,
                values: t.values.clone(),
                basis_values: t.basis_values.clone(),
                fixed_second_component: t.fixed_second_component.clone(),
                raw_normalizer: t.raw_normalizer.clone(),
                approximation_scope: t.approximation_scope.clone(),
            }
        }
        "signed_transform" => {
            if i.reference_jets.is_empty() {
                return Ok(None);
            }
            SignedTransform {
                precision,
                jets: i
                    .reference_jets
                    .iter()
                    .map(|j| SignedJet {
                        ordinal: j.ordinal,
                        t: j.t.clone(),
                        window: j.reference_window.clone(),
                        full: j.reference_full.clone(),
                        tail: j.exterior_tail.clone(),
                        endpoint: j.endpoint_tail_part.clone(),
                        fitted: j.fitted_interior_parts.clone(),
                    })
                    .collect(),
            }
        }
        "arithmetic_energy" => ArithmeticEnergy {
            precision,
            operators: i.components.clone(),
            actions: if i.components.is_empty() {
                run.component_actions.clone()
            } else {
                vec![]
            },
            complete: i.components_are_complete,
            trial: i.target.as_ref().and_then(|t| t.trial_coefficients.clone()),
            deficit: i.deficit.clone(),
            deficit_kind: i.deficit.as_ref().and(i.deficit_kind.clone()),
        },
        "directional_response" => DirectionalResponse {
            precision,
            perturbations: i.perturbations.clone(),
            actions: run.derivative_actions.clone(),
        },
        "weighted_tail" => WeightedTail {
            precision,
            atoms: i.atoms.clone(),
            coordinate: i.atom_coordinate.clone(),
            coverage: i.atom_coverage.clone(),
            checkpoints: i.tail_checkpoints.clone(),
            evaluations: policy.map(|p| p.evaluations.clone()).unwrap_or_default(),
        },
        "spectral_cluster" => SpectralCluster {
            precision,
            cluster: i.cluster.clone(),
            previous_cluster: i.previous_cluster.clone(),
            boundary: i.cluster_boundary_eigenvalues.clone(),
            assessment: if i.cluster.is_empty() {
                i.cluster_preparation_assessment.clone()
            } else {
                None
            },
        },
        "resolution_budget" => {
            if i.reference_jets.is_empty() {
                return Ok(None);
            }
            ResolutionBudget {
                precision,
                jets: i
                    .reference_jets
                    .iter()
                    .map(|j| ResolutionJet {
                        ordinal: j.ordinal,
                        t: j.t.clone(),
                        matched_root_ordinal: j.matched_root_ordinal,
                        error_normalization: j.error_normalization.clone(),
                        source_value_error: j.source_value_error.clone(),
                        tail_value_error: j.tail_value_error.clone(),
                        source_derivative_error: j.source_derivative_error.clone(),
                        root_separation_radius: j.root_separation_radius.clone(),
                    })
                    .collect(),
            }
        }
        "energy_allowance" => {
            let Some(allowance) = &i.energy_allowance else {
                return Ok(None);
            };
            EnergyAllowance {
                precision,
                allowance: allowance.clone(),
            }
        }
        "root_transport" => RootTransport {
            precision,
            perturbations: i.perturbations.clone(),
            actions: run.derivative_actions.clone(),
            log_cutoff_velocity: run.log_cutoff_velocity.clone(),
            checks: c.response_checks.clone(),
        },
        "operator_cluster" => {
            if run.reference_vectors.is_empty() {
                return Ok(None);
            };
            OperatorCluster {
                precision,
                reference_vectors: run.reference_vectors.clone(),
            }
        }
        "finite_section_transfer" => {
            let Some(comparison) = &run.comparison else {
                return Ok(None);
            };
            FiniteSectionTransfer {
                comparison: comparison.clone(),
            }
        }
        "tail_operator" => TailOperator {
            precision,
            form: run.tail_form.clone(),
            atoms: if policy.is_some_and(|p| p.tail_recipe.is_some()) {
                i.atoms.clone()
            } else {
                vec![]
            },
            recipe: policy.and_then(|p| p.tail_recipe.clone()),
            cutoffs: cutoffs(),
            definition: i.definition_digest.clone(),
        },
        "observable_budget" => {
            let Some(uncertainty) = &run.uncertainty else {
                return Ok(None);
            };
            ObservableBudget {
                uncertainty: uncertainty.clone(),
            }
        }
        "capture_preflight" => CapturePreflight {
            sampled_reference: i.target.is_some(),
            reference_jets: !i.reference_jets.is_empty(),
            atoms: !i.atoms.is_empty(),
            independent_actions: !c.independent_actions.is_empty(),
            comparisons: !c.comparisons.is_empty(),
            band: c.band.is_some()
                || run
                    .tail_form
                    .as_ref()
                    .is_some_and(|f| f.polynomial_coordinate.is_some()),
            sector_certificate: c.sector_certificate.is_some(),
            tail_form: run.tail_form.is_some(),
        },
        "consistency" => Consistency {
            precision,
            components: run
                .component_actions
                .iter()
                .filter(|a| a.label == "tau_prime_reconstructed")
                .cloned()
                .collect(),
            independent_actions: c.independent_actions.clone(),
        },
        "configuration_comparison" => {
            if c.comparisons.is_empty() {
                return Ok(None);
            };
            ConfigurationComparison {
                comparisons: c.comparisons.clone(),
            }
        }
        "band_reconstruction" => BandReconstruction {
            precision,
            band: c.band.clone(),
            polynomial_tail: run
                .tail_form
                .as_ref()
                .filter(|f| f.polynomial_coordinate.is_some())
                .cloned(),
            cutoffs: cutoffs(),
        },
        "transform_enclosure" => TransformEnclosure {
            precision,
            contour: c.contour.clone(),
            certificate: c.sector_certificate.clone().map(Box::new),
        },
        _ => bail!("diagnostic has no consumed-input contract"),
    }))
}

pub(super) fn uses_matrix(id: &str) -> bool {
    matches!(
        id,
        "arithmetic_energy"
            | "directional_response"
            | "spectral_cluster"
            | "root_transport"
            | "operator_cluster"
            | "finite_section_transfer"
            | "capture_preflight"
            | "configuration_comparison"
    )
}
pub(super) fn uses_roots(id: &str) -> bool {
    matches!(
        id,
        "directional_response"
            | "resolution_budget"
            | "complex_transform"
            | "root_transport"
            | "observable_budget"
            | "capture_preflight"
            | "configuration_comparison"
            | "transform_enclosure"
    )
}

#[derive(Serialize)]
pub(super) struct NumericalOptions<'a> {
    working_precision_bits: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    exponential_rates: Option<&'a [String]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    maximum_directional_rows: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    relative_tolerance: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reported_working_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reported_output_bytes: Option<u64>,
}
pub(super) fn numerical_options<'a>(
    id: &str,
    o: &'a ExtensionOptions,
    root_count: usize,
) -> NumericalOptions<'a> {
    NumericalOptions {
        working_precision_bits: o.working_precision_bits,
        exponential_rates: (id == "compactness").then_some(o.exponential_rates.as_slice()),
        maximum_directional_rows: (id == "directional_response")
            .then_some(o.maximum_directional_rows.min(root_count)),
        relative_tolerance: (id == "resolution_budget").then_some(o.relative_tolerance.as_str()),
        reported_working_bytes: if id == "capture_preflight" {
            o.maximum_working_bytes
        } else {
            None
        },
        reported_output_bytes: (id == "capture_preflight")
            .then_some(o.maximum_estimated_output_bytes),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> ExternalResearchInputs {
        serde_json::from_value(json!({"schema_version":1,"source_eigenpair":ContentDigest::sha256(b"state"),"lambda_squared":"9","n_modes":0,"precision_bits":128,"convention_id":"provider A","definition_digest":ContentDigest::sha256(b"provider descriptor"),"approximation_scope":"finite fixture","run_once":{"derivative_actions":[{"label":"tau_total","source_digest":ContentDigest::sha256(b"action"),"action":["2"],"convention":"unit state"}]}})).unwrap()
    }
    #[test]
    fn unrelated_external_sections_and_provider_identity_do_not_change_consumed_values() {
        let a = fixture();
        let mut b = a.clone();
        b.definition_digest = ContentDigest::sha256(b"new provider receipt");
        b.convention_id = "provider B".into();
        b.run_once
            .as_mut()
            .unwrap()
            .producer_notes
            .push("another acquisition".into());
        b.run_once.as_mut().unwrap().completion = Some(completion::CompletionInputs {
            preparation_notes: vec!["cohort added later".into()],
            ..Default::default()
        });
        assert!(external_values("compactness", Some(&b)).unwrap().is_none());
        assert!(external_values("complex_transform", Some(&b))
            .unwrap()
            .is_none());
        assert_eq!(
            external_values("directional_response", Some(&a)).unwrap(),
            external_values("directional_response", Some(&b)).unwrap()
        );
        b.run_once.as_mut().unwrap().derivative_actions[0].action[0] = "3".into();
        assert_ne!(
            external_values("directional_response", Some(&a)).unwrap(),
            external_values("directional_response", Some(&b)).unwrap()
        );
        assert!(external_values("unregistered", Some(&a)).is_err());
    }
    #[test]
    fn diagnostic_options_bind_values_and_reported_limits_only() {
        let a = ExtensionOptions {
            working_precision_bits: 192,
            maximum_rows: 100,
            maximum_directional_rows: 10,
            maximum_estimated_output_bytes: 1024 * 1024,
            maximum_working_bytes: Some(1 << 30),
            exponential_rates: vec!["1".into()],
            relative_tolerance: "0.01".into(),
        };
        let mut b = a.clone();
        b.maximum_rows = 200;
        b.maximum_estimated_output_bytes *= 2;
        b.maximum_working_bytes = Some(2 << 30);
        b.relative_tolerance = "0.02".into();
        b.maximum_directional_rows = 20;
        fn key(id: &str, o: &ExtensionOptions) -> ContentDigest {
            xc_cache::canonical_digest(&numerical_options(id, o, 30)).unwrap()
        }
        assert_eq!(key("compactness", &a), key("compactness", &b));
        assert_ne!(key("resolution_budget", &a), key("resolution_budget", &b));
        assert_ne!(
            key("directional_response", &a),
            key("directional_response", &b)
        );
        assert_ne!(key("capture_preflight", &a), key("capture_preflight", &b));
        b = a.clone();
        b.exponential_rates = vec!["2".into()];
        assert_ne!(key("compactness", &a), key("compactness", &b));
        assert_eq!(
            key("directional_response", &a),
            key("directional_response", &b)
        );
    }
}
