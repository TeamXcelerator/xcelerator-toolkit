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
    let l = cutoff.clone().ln();
    let definition = ContentDigest(spec.digest()?);
    let workers = sample_workers(spec, intervals + 1)?;
    let values = sample_values(&target, spec, p, &cutoff, &l, intervals, workers)?;
    let mut samples = Vec::with_capacity(intervals + 1);
    let mut fine = vec![Float::with_val(p, 0); state.modes + 1];
    let mut coarse = fine.clone();
    for (j, value) in values.into_iter().enumerate() {
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
        xc_core::progress_message!("target research checkpoint unavailable: {e}");
    }
    Ok((reference, input))
}

/// Largest default number of concurrent provider processes.
const MAXIMUM_DEFAULT_SAMPLE_WORKERS: usize = 64;

/// Provider processes used for sampling. A provider is evaluated concurrently
/// only when declared pure (`XC_TARGET_PROVIDER_PURE=1`): each value is then
/// the same function of its point in every process. The default is the Rayon
/// worker count, capped at 64; `XC_TARGET_PROVIDER_WORKERS` sets it
/// explicitly (1..=256).
pub(crate) fn sample_workers(
    spec: &crate::target::TargetProfileSpec,
    points: usize,
) -> Result<usize> {
    if spec.external_profile.is_none() || !crate::target::provider_declared_pure()? {
        return Ok(1);
    }
    let workers = match std::env::var("XC_TARGET_PROVIDER_WORKERS") {
        Ok(value) => {
            let workers = value
                .parse::<usize>()
                .map_err(|_| anyhow::anyhow!("XC_TARGET_PROVIDER_WORKERS must be an integer"))?;
            ensure!(
                (1..=256).contains(&workers),
                "XC_TARGET_PROVIDER_WORKERS must be between 1 and 256"
            );
            workers
        }
        Err(std::env::VarError::NotPresent) => {
            rayon::current_num_threads().clamp(1, MAXIMUM_DEFAULT_SAMPLE_WORKERS)
        }
        Err(error) => return Err(error.into()),
    };
    Ok(workers.min(points))
}

/// Target values at `x_j = j*log(cutoff)/(2*intervals)`, `j=0..=intervals`,
/// in index order (see [`target_values_at`]).
fn sample_values(
    target: &crate::target::hp::TargetEvaluator,
    spec: &crate::target::TargetProfileSpec,
    p: u32,
    cutoff: &Float,
    l: &Float,
    intervals: usize,
    workers: usize,
) -> Result<Vec<Float>> {
    let points = (0..=intervals)
        .map(|j| (Float::with_val(p, l) * j / (2 * intervals)).exp())
        .collect::<Vec<_>>();
    target_values_at(target, spec, p, &cutoff.clone().sqrt(), &points, workers)
}

/// Target values at `points`, in order. With several workers, each evaluates
/// one contiguous range of points through its own provider process. The
/// values, and the sums a caller forms from them in point order, are identical
/// to sequential evaluation; an error is reported for the lowest failing point,
/// as sequentially.
pub(crate) fn target_values_at(
    target: &crate::target::hp::TargetEvaluator,
    spec: &crate::target::TargetProfileSpec,
    p: u32,
    lambda: &Float,
    points: &[Float],
    workers: usize,
) -> Result<Vec<Float>> {
    if workers <= 1 || points.len() <= 1 {
        return points.iter().map(|u| target.try_value(u)).collect();
    }
    let chunk = points.len().div_ceil(workers);
    let parts = std::thread::scope(|scope| {
        let handles = points
            .chunks(chunk)
            .map(|part| {
                scope.spawn(move || -> Result<Vec<Float>> {
                    let worker = crate::target::hp::TargetEvaluator::from_spec(spec, p)?;
                    worker.validate_lambda(lambda)?;
                    part.iter().map(|u| worker.try_value(u)).collect()
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .unwrap_or_else(|_| Err(anyhow::anyhow!("target sampling worker panicked")))
            })
            .collect::<Vec<_>>()
    });
    let mut values = Vec::with_capacity(points.len());
    for part in parts {
        values.extend(part?);
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concurrent_pure_provider_sampling_equals_sequential_sampling() {
        use std::{fs, process::Command};
        let root_dir = xc_core::test_support::TestDir::new("parallel-target-samples");
        let root = root_dir.to_path_buf();
        let source = root.join("provider.rs");
        // Replies with the requested point itself: distinct, deterministic values.
        fs::write(&source, r#"
use std::io::{self,BufRead,Write};
fn field<'a>(line:&'a str,name:&str)->Option<&'a str>{
 let marker=format!("\"{}\":",name);
 Some(line.split_once(&marker)?.1.split([',','}']).next().unwrap())
}
fn main(){
 for line in io::stdin().lock().lines(){
  let line=line.unwrap();
  let id=field(&line,"request_id").unwrap();let nonce=field(&line,"request_nonce").unwrap();
  let operation=field(&line,"operation").unwrap();let bits=field(&line,"precision_bits").unwrap();
  let value=field(&line,"u").unwrap_or("\"1\"");
  println!("{{\"protocol_version\":2,\"request_id\":{id},\"request_nonce\":{nonce},\"operation\":{operation},\"precision_bits\":{bits},\"ready\":true,\"value\":{value}}}");
  io::stdout().flush().unwrap();
 }
}
"#).unwrap();
        let executable = root.join("provider");
        let output = Command::new("rustc")
            .arg("--edition=2021")
            .arg(&source)
            .arg("-o")
            .arg(&executable)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let digest = ContentDigest::sha256(&fs::read(&executable).unwrap()).0;
        let previous = std::env::var_os("XC_TARGET_PROVIDER_EXECUTABLE");
        std::env::set_var("XC_TARGET_PROVIDER_EXECUTABLE", &executable);
        let spec = crate::target::TargetProfileSpec::from_json(
            &serde_json::to_vec(&serde_json::json!({
                "schema_version":3,"profile_id":"parallel-samples","external_profile":{
                    "lambda_squared":"13","evaluation_precision_bits":320,"provider_sha256":digest,"input":{}
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let p = 256;
        let cutoff = Float::with_val(p, 13);
        let l = cutoff.clone().ln();
        let target = crate::target::hp::TargetEvaluator::from_spec(&spec, p).unwrap();
        let sequential = sample_values(&target, &spec, p, &cutoff, &l, 64, 1);
        let concurrent = sample_values(&target, &spec, p, &cutoff, &l, 64, 7);
        match previous {
            Some(value) => std::env::set_var("XC_TARGET_PROVIDER_EXECUTABLE", value),
            None => std::env::remove_var("XC_TARGET_PROVIDER_EXECUTABLE"),
        }
        let (sequential, concurrent) = (sequential.unwrap(), concurrent.unwrap());
        assert_eq!(sequential.len(), 65);
        assert_eq!(sequential, concurrent);
        assert!(sequential.windows(2).all(|pair| pair[0] < pair[1]));
    }
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
            "producer_toolkit_version":xc_cache::ToolkitVersion::parse("0.16.0").unwrap(),
            "minimum_reader_version":xc_cache::ToolkitVersion::parse("0.16.0").unwrap(),
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
