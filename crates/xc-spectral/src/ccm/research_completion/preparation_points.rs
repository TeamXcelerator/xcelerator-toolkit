//! Preserve point ownership when preparing a higher-precision input bundle.
//! Metadata and independently precision-tagged comparisons/certificates are unchanged.
use super::{CompletionInputs, TailFormRecipe};
use crate::ccm::{
    convergence_capture::TailForm,
    extended_research::{SampledReference, WeightedAtom},
    retained_evidence::scalar,
};
use anyhow::Result;
use rug::Float;
use xc_numerics::prefix::lossless_decimal;

pub(super) struct Promotion {
    pub source: u32,
    pub target: u32,
}
impl Promotion {
    fn value(&self, value: &mut String) -> Result<()> {
        // Serialize at TARGET precision: serializing at the old precision and
        // reparsing that decimal at the new precision would invent extra bits.
        *value = lossless_decimal(&Float::with_val(self.target, scalar(value, self.source)?));
        Ok(())
    }
    fn upper(&self, value: &mut String) -> Result<()> {
        let bound = crate::ccm::retained_evidence::finite_math::decimal_upper(value, self.source)?;
        *value = Float::with_val_round(self.target, bound, rug::float::Round::Up)
            .0
            .to_string_radix_round(10, None, rug::float::Round::Up);
        Ok(())
    }
    fn values(&self, values: &mut [String]) -> Result<()> {
        for value in values {
            self.value(value)?;
        }
        Ok(())
    }
    fn optional(&self, value: &mut Option<String>) -> Result<()> {
        if let Some(value) = value {
            self.value(value)?;
        }
        Ok(())
    }
    pub(super) fn sampled(&self, sample: &mut SampledReference) -> Result<()> {
        let SampledReference {
            definition_digest: _,
            evaluation_policy: _,
            approximation_scope: _,
            intervals: _,
            values,
            basis_values,
            fixed_second_component,
            raw_normalizer,
            trial_coefficients,
        } = sample;
        self.values(values)?;
        for values in basis_values {
            self.values(values)?;
        }
        self.optional(fixed_second_component)?;
        self.value(raw_normalizer)?;
        if let Some(values) = trial_coefficients {
            self.values(values)?;
        }
        Ok(())
    }
    pub(super) fn atoms(&self, atoms: &mut [WeightedAtom]) -> Result<()> {
        for WeightedAtom {
            ordinal: _,
            coordinate,
            weight,
            family: _,
            partition: _,
        } in atoms
        {
            self.value(coordinate)?;
            self.value(weight)?;
        }
        Ok(())
    }
    pub(super) fn form(&self, form: &mut TailForm) -> Result<()> {
        let TailForm {
            polynomial_coordinate: _,
            definition_digest: _,
            dimension: _,
            finite_zero_form,
            tail_correction,
            lattice_gram,
            tail_operator_error,
            coverage: _,
            hypotheses: _,
        } = form;
        for values in [finite_zero_form, tail_correction, lattice_gram] {
            self.values(values)?;
        }
        if let Some(bound) = tail_operator_error {
            self.upper(bound)?;
        }
        Ok(())
    }
    pub(super) fn recipe(&self, recipe: &mut TailFormRecipe) -> Result<()> {
        let TailFormRecipe {
            basis_polynomials,
            tail_correction,
            hypotheses: _,
        } = recipe;
        for values in basis_polynomials {
            self.values(values)?;
        }
        if let Some(values) = tail_correction {
            self.values(values)?;
        }
        Ok(())
    }
    pub(super) fn completion(&self, completion: &mut CompletionInputs) -> Result<()> {
        let CompletionInputs {
            atom_analysis,
            independent_actions,
            response_checks,
            comparisons: _,
            band,
            contour,
            sector_certificate: _,
            preparation_notes: _,
        } = completion;
        // Comparisons carry their own state/root precision. Portable certificates
        // carry directed bounds and provenance; neither inherits the bundle p.
        if let Some(policy) = atom_analysis {
            let crate::ccm::atom_research::AtomAnalysisPolicy {
                maximum_atoms: _,
                maximum_input_bytes: _,
                weighted_chunks: _,
                band_chunks: _,
                evaluations,
                cutoffs,
                tail_recipe,
            } = policy;
            self.values(cutoffs)?;
            for crate::ccm::atom_research::AtomEvaluation {
                ordinal: _,
                coordinate,
                label: _,
                exclude: _,
            } in evaluations
            {
                self.value(coordinate)?;
            }
            if let Some(recipe) = tail_recipe {
                self.recipe(recipe)?;
            }
        }
        for crate::ccm::convergence_capture::OperatorAction {
            label: _,
            source_digest: _,
            action,
            convention: _,
        } in independent_actions
        {
            self.values(action)?;
        }
        for super::ResponseCheck {
            ordinal: _,
            t,
            source_digest: _,
            branch: _,
            coordinate: _,
            derivative_parameter: _,
            activation_convention: _,
            fixed_velocity,
            support_velocity,
            total_velocity,
        } in response_checks
        {
            self.value(t)?;
            for value in [fixed_velocity, support_velocity, total_velocity] {
                self.optional(value)?;
            }
        }
        if let Some(band) = band {
            let super::SignedBandModel {
                degree: _,
                coordinate: _,
                definition_digest: _,
                atoms,
                coverage: _,
                hypotheses: _,
                borrowed_inputs: _,
                input_energy,
                scoring_roots,
            } = band;
            for super::BandAtom {
                coordinate,
                signed_weight,
                family: _,
            } in atoms
            {
                self.value(coordinate)?;
                self.value(signed_weight)?;
            }
            self.optional(input_energy)?;
            self.values(scoring_roots)?;
        }
        if let Some(contour) = contour {
            let super::ContourPolicy {
                left,
                right,
                bottom,
                top,
                maximum_depth: _,
                maximum_segments: _,
            } = contour;
            for value in [left, right, bottom, top] {
                self.value(value)?;
            }
        }
        Ok(())
    }
}

impl Promotion {
    fn jet(&self, jet: &mut crate::ccm::extended_research::Jet) -> Result<()> {
        self.value(&mut jet.value)?;
        self.value(&mut jet.derivative)
    }
    fn components(
        &self,
        parts: &mut [crate::ccm::extended_research::OperatorComponent],
    ) -> Result<()> {
        for crate::ccm::extended_research::OperatorComponent {
            label: _,
            source_digest: _,
            diagonal,
            dense,
            rank_one,
        } in parts
        {
            self.values(diagonal)?;
            self.values(dense)?;
            for crate::ccm::extended_research::RankOneTerm { weight, vector } in rank_one {
                self.value(weight)?;
                self.values(vector)?;
            }
        }
        Ok(())
    }
    fn run_once(&self, run: &mut crate::ccm::convergence_capture::RunOnceInputs) -> Result<()> {
        let crate::ccm::convergence_capture::RunOnceInputs {
            completion,
            component_actions,
            derivative_actions,
            log_cutoff_velocity,
            reference_vectors,
            comparison: _,
            tail_form,
            uncertainty,
            producer_notes: _,
        } = run;
        // A comparison owns its own precision; it does not inherit the bundle's.
        if let Some(completion) = completion {
            self.completion(completion)?;
        }
        for action in component_actions.iter_mut().chain(derivative_actions) {
            self.values(&mut action.action)?;
        }
        self.optional(log_cutoff_velocity)?;
        for vector in reference_vectors {
            self.values(vector)?;
        }
        if let Some(form) = tail_form {
            self.form(form)?;
        }
        if let Some(uncertainty) = uncertainty {
            self.upper(&mut uncertainty.unit_state_l2_error)?;
        }
        Ok(())
    }
    pub(super) fn bundle(
        &self,
        input: &mut crate::ccm::extended_research::ExternalResearchInputs,
    ) -> Result<()> {
        let crate::ccm::extended_research::ExternalResearchInputs {
            run_once,
            schema_version: _,
            source_eigenpair: _,
            lambda_squared: _,
            n_modes: _,
            precision_bits: _,
            convention_id: _,
            definition_digest: _,
            approximation_scope: _,
            target,
            reference_jets,
            components,
            components_are_complete: _,
            perturbations,
            deficit,
            deficit_kind: _,
            atoms,
            atom_coordinate: _,
            atom_coverage: _,
            tail_checkpoints,
            cluster: _,
            previous_cluster: _,
            cluster_boundary_eigenvalues,
            energy_allowance,
        } = input;
        // Cluster vectors own separate precisions. The cutoff is an exact decimal
        // parameter, so neither is reinterpreted as an inherited numerical point.
        if let Some(run) = run_once {
            self.run_once(run)?;
        }
        if let Some(target) = target {
            self.sampled(target)?;
        }
        for crate::ccm::extended_research::ReferenceJet {
            matched_root_ordinal: _,
            ordinal: _,
            t,
            reference_window,
            reference_full,
            exterior_tail,
            endpoint_tail_part,
            fitted_interior_parts,
            error_normalization: _,
            source_value_error,
            tail_value_error,
            source_derivative_error,
            root_separation_radius,
        } in reference_jets
        {
            self.value(t)?;
            for jet in [reference_window, reference_full, exterior_tail] {
                self.jet(jet)?;
            }
            if let Some(jet) = endpoint_tail_part {
                self.jet(jet)?;
            }
            for jet in fitted_interior_parts {
                self.jet(jet)?;
            }
            for value in [
                source_value_error,
                tail_value_error,
                source_derivative_error,
                root_separation_radius,
            ] {
                self.optional(value)?;
            }
        }
        self.components(components)?;
        self.components(perturbations)?;
        self.optional(deficit)?;
        self.atoms(atoms)?;
        self.values(tail_checkpoints)?;
        if let Some(boundary) = cluster_boundary_eigenvalues {
            self.values(boundary)?;
        }
        if let Some(crate::ccm::extended_research::EnergyAllowance {
            upper_trial_energy,
            low_block_lower_bound,
            high_block_lower_bound,
            cross_block_norm_bound,
            hypothesis_record_digest: _,
            hypotheses: _,
        }) = energy_allowance
        {
            for value in [
                upper_trial_energy,
                low_block_lower_bound,
                high_block_lower_bound,
                cross_block_norm_bound,
            ] {
                self.value(value)?;
            }
        }
        Ok(())
    }
}
