//! Numeric research inputs from the explicitly configured runtime target.
//! Only samples and their finite Fourier projection are prepared here. No
//! infinite transform jets, tail model, atoms, or theorem bounds are inferred.
use super::{extended_research::*, retained_evidence::*, state_geometry::RetainedState};
use anyhow::{ensure, Result};
use rug::{float::Constant, Float};
use xc_cache::ContentDigest;
use xc_numerics::prefix::lossless_decimal as dec;

pub(super) fn prepare(
    state: &RetainedState,
    spec: &crate::target::TargetProfileSpec,
    intervals: usize,
) -> Result<(ResearchInputs, ExternalResearchInputs)> {
    ensure!(
        intervals >= 2 * state.modes + 2 && intervals <= 131072 && intervals.is_multiple_of(2),
        "target research sampling requires an even grid resolving the retained Fourier modes"
    );
    let _stage = super::capture_runtime::Stage::new("runtime target research samples");
    let p = state.precision.saturating_add(64);
    precision(p)?;
    let store = super::capture_runtime::Checkpoints::new(&(
        "runtime-target-samples-and-fourier-v1",
        &state.manifest.content_digest,
        &state.cutoff,
        state.modes,
        p,
        intervals,
        spec.digest()?,
    ))?;
    if let Some((reference, input)) =
        store.load::<(ResearchInputs, ExternalResearchInputs)>("reference")?
    {
        reference.validate()?;
        input.matches(state)?;
        return Ok((reference, input));
    }
    let target = crate::target::hp::TargetEvaluator::from_spec(spec, p)?;
    let cutoff = scalar(&state.cutoff, p)?;
    target.validate_lambda(&cutoff.clone().sqrt())?;
    let l = cutoff.ln();
    let definition = ContentDigest(spec.digest()?);
    let mut samples = Vec::with_capacity(intervals + 1);
    let mut fine = vec![Float::with_val(p, 0); state.modes + 1];
    let mut coarse = fine.clone();
    for j in 0..=intervals {
        let x = Float::with_val(p, &l) * j / (2 * intervals);
        let value = target.try_value(&x.exp())?;
        let angle = Float::with_val(p, Constant::Pi) * (Float::with_val(p, j) / intervals + 1u32);
        let cosine = angle.cos();
        let mut previous = Float::with_val(p, 1);
        let mut current = cosine.clone();
        let weight = if j == 0 || j == intervals { 2u32 } else { 1u32 };
        for k in 0..=state.modes {
            let basis = if k == 0 {
                Float::with_val(p, 1)
            } else {
                current.clone()
            };
            let contribution = Float::with_val(p, &value) * basis / weight;
            fine[k] += &contribution;
            if j.is_multiple_of(2) {
                coarse[k] += &contribution;
            }
            if k > 0 {
                let next = Float::with_val(p, &cosine) * &current * 2u32 - &previous;
                previous = current;
                current = next;
            }
        }
        samples.push(dec(&value));
    }
    let mut refinement = Float::with_val(p, 0);
    for (f, c) in fine.iter_mut().zip(&mut coarse) {
        *f /= intervals;
        *c /= intervals / 2;
        refinement = refinement.max(&(Float::with_val(p, &*f) - &*c).abs());
    }
    let coefficients = (0..=2 * state.modes)
        .map(|j| dec(&fine[j.abs_diff(state.modes)]))
        .collect();
    let scope = format!("supplied runtime target sampled on {intervals} uniform log-coordinate intervals; finite even Fourier projection through mode {}; composite trapezoid; maximum coefficient change on half grid={}; this measured refinement is not a quadrature, source-error, or infinite-tail certificate", state.modes, dec(&refinement));
    let reference = ResearchInputs {
        schema_version: 1,
        reference: ReferenceSpec {
            schema_version: 1,
            definition: format!(
                "finite Fourier projection of runtime target {}",
                definition.0
            ),
            lambda_squared: state.cutoff.clone(),
            precision_bits: p,
            coefficients,
            approximation_scope: scope.clone(),
        },
        basis: vec![],
        projection: ProjectionOptions {
            working_precision_bits: p + 64,
            normalization: "center_one".into(),
            fixed_second_component: None,
        },
    };
    reference.validate()?;
    let mut input: ExternalResearchInputs = serde_json::from_value(serde_json::json!({
        "schema_version":1,"source_eigenpair":state.manifest.content_digest,
        "lambda_squared":state.cutoff,"n_modes":state.modes,"precision_bits":p,
        "convention_id":"runtime-target-samples-and-fourier-v1",
        "definition_digest":definition,"approximation_scope":scope
    }))?;
    input.target = Some(SampledReference {
        definition_digest: definition,
        evaluation_policy: "configured target evaluator; uniform x=log(u) nodes; normalized at u=1"
            .into(),
        approximation_scope: scope,
        intervals,
        values: samples,
        basis_values: vec![],
        fixed_second_component: None,
        raw_normalizer: "1".into(),
        trial_coefficients: Some(reference.reference.coefficients.clone()),
    });
    input.validate()?;
    if let Err(e) = store.save("reference", &(&reference, &input)) {
        eprintln!("target research checkpoint unavailable: {e}");
    }
    Ok((reference, input))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn supplied_target_preparation_preserves_identity_and_does_not_invent_tail_inputs() {
        use crate::target::{GaussianPolynomialSeriesSpec, ScalarScaleSpec, TargetProfileSpec};
        let spec = TargetProfileSpec {
            schema_version: 1,
            profile_id: "research-sample-fixture".into(),
            external_profile: None,
            auxiliary_series: None,
            base_series: Some(GaussianPolynomialSeriesSpec {
                term_input_power: 0,
                polynomial_coefficients: vec!["1".into()],
                polynomial_scale: ScalarScaleSpec::default(),
                parameter_polynomial_coefficients: vec![],
                parameter_polynomial_scale: ScalarScaleSpec::default(),
                minimum_terms: 2,
                maximum_terms: 1000,
            }),
        };
        let digest = ContentDigest::sha256(b"sample-state");
        let manifest = serde_json::from_value(serde_json::json!({
            "schema_version":1,"key":xc_cache::ArtifactKey::new("ccm_weil_eigenpair","test",b"test").unwrap(),
            "content_digest":digest,"size_bytes":12,
            "objects":[{"content_digest":digest,"size_bytes":12}],"created_unix_seconds":1,
            "producer_toolkit_version":xc_cache::ToolkitVersion::parse("0.15.1").unwrap(),
            "minimum_reader_version":xc_cache::ToolkitVersion::parse("0.15.1").unwrap(),
            "maximum_reader_version":null,"quality":"validated","visibility":"private",
            "immutable":true,"dependencies":[],"tags":{},"provenance_digest":null
        })).unwrap();
        let state = RetainedState {
            manifest,
            cutoff: "9".into(),
            modes: 2,
            precision: 128,
            coefficients: vec![Float::with_val(128, 1); 5],
            eigenvalue: "1".into(),
            selection_policy: None,
        };
        let (reference, input) = prepare(&state, &spec, 256).unwrap();
        assert_eq!(input.source_eigenpair, state.manifest.content_digest);
        assert_eq!(input.definition_digest.0, spec.digest().unwrap());
        assert_eq!(reference.reference.coefficients.len(), 2 * state.modes + 1);
        assert_eq!(input.target.as_ref().unwrap().values.len(), 257);
        assert_eq!(
            scalar(&input.target.as_ref().unwrap().values[0], 192).unwrap(),
            1
        );
        assert!(input.reference_jets.is_empty());
        assert!(input.atoms.is_empty());
        assert!(input.energy_allowance.is_none());
        assert!(reference
            .reference
            .approximation_scope
            .contains("not a quadrature"));
        assert!(prepare(&state, &spec, 255).is_err());
    }
}
