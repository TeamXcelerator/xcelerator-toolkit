//! Derive finite research inputs from the retained run and the bundled ordinate table.
//! No result from another run, fitted energy, or assumed ordinate-tail constant is used.
use super::{
    convergence_capture::TailForm, extended_research::*, retained_evidence::*,
    state_geometry::RetainedState,
};
use anyhow::{ensure, Result};
use rayon::prelude::*;
use rug::{float::Constant, ops::Pow, Float};
use xc_cache::ContentDigest;
use xc_numerics::prefix::lossless_decimal as dec;

/// Prepare weighted atoms, a finite arithmetic tail form, and Fourier block bounds.
///
/// The matrix and state must be authenticated retained parents. The ordinate decimals
/// are explicit comparison geometry. The arithmetic completion includes the entire
/// retained form; interpreting it as an unknown critical-line zero sum is not assumed.
/// Values are point diagnostics of the stored section, not continuum certificates.
pub fn prepare_arithmetic_inputs(
    state: &RetainedState,
    matrix: &RetainedMatrix<'_>,
) -> Result<ExternalResearchInputs> {
    matrix.match_state(state)?;
    ensure!(state.modes > 0 && state.modes <= 512, "automatic polynomial preparation supports 1..512 modes; larger sections require an explicit resource policy");
    let _stage = super::capture_runtime::Stage::new("retained arithmetic research preparation");
    let p = state.precision.saturating_add(128);
    precision(p)?;
    let table = xc_zeta::zeros::bundled_dataset_identity()?;
    let definition = ContentDigest::sha256(&serde_json::to_vec(&(
        "retained-arithmetic-polynomial-v2",
        &matrix.manifest.content_digest,
        &table,
        state.modes,
        p,
    ))?);
    let store =
        super::capture_runtime::Checkpoints::new(&(&definition, &state.manifest.content_digest))?;
    if let Some(input) = store.load::<ExternalResearchInputs>("arithmetic-inputs")? {
        input.matches(state)?;
        return Ok(input);
    }
    let n = state.modes;
    let dim = 2 * n + 1;
    let l = scalar(&state.cutoff, p)?.ln();
    let beta = Float::with_val(p, Constant::Pi) * 2u32 / &l;
    let scale = Float::with_val(p, &beta) * n;
    let zeros = xc_zeta::zeros::bundled_first_n_strings(table.record_count)?;
    let ordinates = coeffs(&zeros, p)?;
    let z = ordinates
        .iter()
        .map(|t| (Float::with_val(p, t) / &scale).square())
        .collect::<Vec<_>>();
    // Fixed before any solve: retain at most degree 64 after a known-ordinate prefix.
    let degree = n.min(64);
    let prefix = n - degree;
    let d = degree + 1;
    let scope = format!("finite retained arithmetic section; coordinate z=(t/(2*pi*N/log(C)))^2; first {prefix} bundled ordinate decimals fixed as prefix; monomial basis 1,z,...,z^{degree} after that prefix; all {} bundled atoms through t={}; table sha256={}; arithmetic remainder=V^T*A*V/2-explicit head, includes all omitted signed form contributions and assembly error; no RH, complete critical-line head, continuum transfer, or source-accuracy premise", table.record_count, zeros.last().unwrap(), table.content_sha256);
    let mut input: ExternalResearchInputs = serde_json::from_value(serde_json::json!({
        "schema_version":1,"source_eigenpair":state.manifest.content_digest,
        "lambda_squared":state.cutoff,"n_modes":n,"precision_bits":p,
        "convention_id":"retained-arithmetic-polynomial-v2","definition_digest":definition,
        "approximation_scope":scope
    }))?;
    // State-weighted measures: transform squared and physical unit coefficient masses.
    input.atoms = ordinates
        .par_iter()
        .enumerate()
        .map(|(j, t)| -> Result<_> {
            let (v, _, _, _) = transform_terms(state, t, p)?;
            Ok(WeightedAtom {
                ordinal: j + 1,
                coordinate: dec(&z[j]),
                weight: dec(&v.square()),
                family: "zero".into(),
                partition: "bundled_decimal_head".into(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let unit = source_unit(state, p);
    for k in 0..=n {
        let mass = if k == 0 {
            unit[n].clone().square()
        } else {
            unit[n - k].clone().square() + unit[n + k].clone().square()
        };
        input.atoms.push(WeightedAtom {
            ordinal: k + 1,
            coordinate: dec(&(Float::with_val(p, k) / n).square()),
            weight: dec(&mass),
            family: "lattice".into(),
            partition: "all_modes_including_origin;ordinal=mode+1".into(),
        });
    }
    input.atom_coordinate = Some("z=(t/(2*pi*N/log(C)))^2; lattice z=(mode/N)^2".into());
    input.atom_coverage = Some(scope.clone());

    // V in the full -N..N coefficient coordinates. The two reflected entries
    // each equal (-1)^k (k_k/kappa) F(a_k)/sqrt(L), so their norm includes d_k.
    let mut factorial = vec![Float::with_val(p, 1); 2 * n + 1];
    for k in 1..=2 * n {
        factorial[k] = Float::with_val(p, &factorial[k - 1]) * k;
    }
    let n_power = Float::with_val(p, n).pow(2 * n as u32);
    let mut v = vec![Float::with_val(p, 0); dim * d];
    for k in 0..=n {
        let a = (Float::with_val(p, k) / n).square();
        let mut value =
            Float::with_val(p, &n_power) / &factorial[n - k] / &factorial[n + k] / l.clone().sqrt();
        if k % 2 == 1 {
            value = -value;
        }
        for r in &z[..prefix] {
            value *= Float::with_val(p, &a) - r;
        }
        for j in 0..d {
            v[(n + k) * d + j] = value.clone();
            v[(n - k) * d + j] = value.clone();
            value *= &a;
        }
    }
    let av = (0..dim)
        .into_par_iter()
        .map(|r| {
            (0..d)
                .map(|j| {
                    let mut sum = Float::with_val(p, 0);
                    for k in 0..dim {
                        // The quadratic form of a slightly asymmetric retained point matrix
                        // is exactly the form of its symmetric part.
                        let a = (Float::with_val(p, &matrix.entries[r * dim + k])
                            + &matrix.entries[k * dim + r])
                            / 2u32;
                        sum += a * &v[k * d + j];
                    }
                    sum
                })
                .collect::<Vec<_>>()
        })
        .flatten()
        .collect::<Vec<_>>();
    let mut gram = vec![Float::with_val(p, 0); d * d];
    let mut actual = gram.clone();
    for i in 0..d {
        for j in 0..=i {
            let mut g = Float::with_val(p, 0);
            let mut a = g.clone();
            for k in 0..dim {
                g += Float::with_val(p, &v[k * d + i]) * &v[k * d + j];
                a += Float::with_val(p, &v[k * d + i]) * &av[k * d + j];
            }
            gram[i * d + j] = g.clone();
            gram[j * d + i] = g;
            a /= 2u32;
            actual[i * d + j] = a.clone();
            actual[j * d + i] = a;
        }
    }
    let mut moments = vec![Float::with_val(p, 0); 2 * d - 1];
    for (j, t) in ordinates.iter().enumerate().skip(prefix) {
        let x = Float::with_val(p, t) / &beta;
        let pi_x = Float::with_val(p, Constant::Pi) * &x;
        let mut f = pi_x.clone().sin() / pi_x;
        for r in &z[..prefix] {
            f *= Float::with_val(p, &z[j]) - r;
        }
        for k in 1..=n {
            f /= Float::with_val(p, &z[j]) - (Float::with_val(p, k) / n).square();
        }
        let mut weight = f.square();
        for m in &mut moments {
            *m += &weight;
            weight *= &z[j];
        }
    }
    let head = (0..d * d)
        .map(|k| moments[k / d + k % d].clone())
        .collect::<Vec<_>>();
    let tail = actual
        .iter()
        .zip(&head)
        .map(|(a, h)| Float::with_val(p, a) - h)
        .collect::<Vec<_>>();
    let form=TailForm {definition_digest:definition.clone(),dimension:d,finite_zero_form:head.iter().map(dec).collect(),tail_correction:tail.iter().map(dec).collect(),lattice_gram:gram.iter().map(dec).collect(),tail_operator_error:None,coverage:scope.clone(),hypotheses:vec!["Point arithmetic of the exact retained parent; assembly/source error is not bounded here. Model energy is computed without retained eigenvalue or eigenvector inputs.".into()],polynomial_coordinate:Some(input.atom_coordinate.clone().unwrap())};
    input.run_once = Some(super::convergence_capture::RunOnceInputs {
        tail_form: Some(form),
        ..Default::default()
    });
    input.energy_allowance = Some(block_allowance(state, matrix, p)?);
    input.validate()?;
    if let Err(e) = store.save("arithmetic-inputs", &input) {
        eprintln!("arithmetic preparation checkpoint unavailable: {e}");
    }
    Ok(input)
}

// Roundtrip decimals can lie on either side of the binary value when parsed
// at a higher precision. One outward ULP preserves a scalar bound on export.
fn bound_decimal(value: &Float, upper: bool) -> String {
    let mut outward = value.clone();
    if outward != 0 {
        if upper {
            outward.next_up();
        } else {
            outward.next_down();
        }
    }
    dec(&outward)
}
fn block_allowance(s: &RetainedState, m: &RetainedMatrix<'_>, p: u32) -> Result<EnergyAllowance> {
    use xc_numerics::mpfr_interval::MpfrInterval as I;
    let _stage = super::capture_runtime::Stage::new("finite matrix block bounds");
    let n = s.modes;
    let d = 2 * n + 1;
    let split = n / 2;
    let low = (0..d)
        .filter(|k| k.abs_diff(n) <= split)
        .collect::<Vec<_>>();
    let high = (0..d).filter(|k| k.abs_diff(n) > split).collect::<Vec<_>>();
    let half = I::point(Float::with_val(p, 0.5));
    let a = |i: usize, j: usize| {
        I::point(Float::with_val(p, &m.entries[i * d + j]))
            .add(&I::point(Float::with_val(p, &m.entries[j * d + i])))
            .mul(&half)
    };
    let mut notes = Vec::new();
    let mut lower = |indices: &[usize]| -> Result<Float> {
        let size = indices.len();
        let mut block = Vec::with_capacity(size * size);
        for &i in indices {
            for &j in indices {
                block.push(a(i, j));
            }
        }
        let gersh = (0..size)
            .map(|i| {
                let mut v = block[i * size + i].clone();
                for j in 0..size {
                    if i != j {
                        let radius = block[i * size + j]
                            .lower()
                            .clone()
                            .abs()
                            .max(&block[i * size + j].upper().clone().abs());
                        v = v.sub(&I::point(radius));
                    }
                }
                v.lower().clone()
            })
            .min_by(Float::total_cmp)
            .unwrap();
        let point = block
            .iter()
            .map(|a| a.midpoint_point().lower().clone())
            .collect::<Vec<_>>();
        let values = xc_numerics::eigen::dense_symmetric_eigenvalues_hp_stable(&point, size, p);
        if let Ok(values) = values {
            let min = values.into_iter().min_by(Float::total_cmp).unwrap();
            let scale = point
                .iter()
                .map(|a| a.clone().abs())
                .max_by(Float::total_cmp)
                .unwrap()
                + 1u32;
            let candidate = Float::with_val(p, &min) - min.abs() / 16u32 - (scale >> 128u32);
            if candidate > gersh {
                let shift = I::point(candidate.clone());
                for j in 0..size {
                    block[j * size + j] = block[j * size + j].sub(&shift);
                }
                let mut minimum: Option<Float> = None;
                let mut positive = true;
                for j in 0..size {
                    let pivot = block[j * size + j].clone();
                    if !pivot.is_strictly_positive() {
                        positive = false;
                        break;
                    }
                    minimum = Some(
                        minimum.map_or_else(|| pivot.lower().clone(), |x| x.min(pivot.lower())),
                    );
                    for i in j + 1..size {
                        let factor = block[i * size + j].div(&pivot)?;
                        for k in i..size {
                            block[k * size + i] =
                                block[k * size + i].sub(&block[k * size + j].mul(&factor));
                        }
                    }
                }
                if positive {
                    notes.push(format!("{size}-dimensional shifted block: all interval LDL pivots strictly positive; minimum pivot lower endpoint={}",dec(&minimum.unwrap())));
                    return Ok(candidate);
                }
            }
        }
        notes.push(format!("{size}-dimensional block uses outward Gershgorin lower bound; sharper interval LDL bound unresolved"));
        Ok(gersh)
    };
    let b = lower(&low)?;
    let mu = lower(&high)?;
    let mut cross = I::from_i64(0, p);
    for &i in &low {
        for &j in &high {
            cross = cross.add(&a(i, j).square());
        }
    }
    // Use the raw dyadic state and divide by its interval norm. No assumption
    // that a rounded normalized vector has exact norm one is necessary.
    let v = s
        .coefficients
        .iter()
        .map(|x| I::point(Float::with_val(p, x)))
        .collect::<Vec<_>>();
    let norm = v
        .iter()
        .fold(I::from_i64(0, p), |sum, x| sum.add(&x.square()));
    let mut energy = I::from_i64(0, p);
    for i in 0..d {
        for j in 0..d {
            energy = energy.add(&a(i, j).mul(&v[i]).mul(&v[j]));
        }
    }
    let trial = energy.div(&norm)?;
    let cross = cross.sqrt()?;
    notes.push(format!("outward MPFR bounds on retained symmetric Fourier form {}; low |k|<={split}, high {split}<|k|<={n}; interval LDL sharpens Gershgorin when verified; cross-block Frobenius upper endpoint; trial Rayleigh upper endpoint; applies to exact stored point matrix only, with no assembly-error or infinite omitted-mode certificate",m.manifest.content_digest.0));
    Ok(EnergyAllowance {
        upper_trial_energy: bound_decimal(trial.upper(), true),
        low_block_lower_bound: bound_decimal(&b, false),
        high_block_lower_bound: bound_decimal(&mu, false),
        cross_block_norm_bound: bound_decimal(cross.upper(), true),
        hypothesis_record_digest: ContentDigest::sha256(&serde_json::to_vec(&notes)?),
        hypotheses: notes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use xc_cache::*;

    #[test]
    fn exported_scalar_bounds_remain_outward_at_higher_reader_precision() {
        for text in ["0", "0.1", "-0.1", "1e-60", "-1e-60"] {
            let value = scalar(text, 192).unwrap();
            assert!(scalar(&bound_decimal(&value, true), 512).unwrap() >= value);
            assert!(scalar(&bound_decimal(&value, false), 512).unwrap() <= value);
        }
    }

    fn manifest(kind: &str, bytes: &[u8]) -> ArtifactManifest {
        let digest = ContentDigest::sha256(bytes);
        serde_json::from_value(serde_json::json!({
            "schema_version":1,"key":ArtifactKey::new(kind,"preparation-test",bytes).unwrap(),
            "content_digest":digest,"size_bytes":bytes.len(),"objects":[{"content_digest":digest,"size_bytes":bytes.len()}],
            "created_unix_seconds":1,"producer_toolkit_version":ToolkitVersion::parse("0.15.1").unwrap(),
            "minimum_reader_version":ToolkitVersion::parse("0.15.1").unwrap(),"maximum_reader_version":null,
            "quality":"validated","visibility":"private","immutable":true,"dependencies":[],"tags":{},"provenance_digest":null
        })).unwrap()
    }

    #[test]
    fn arithmetic_completion_preserves_identity_form_and_origin_mass() {
        let p = 256;
        let n = 3;
        let d = 2 * n + 1;
        let s = RetainedState {
            manifest: manifest("ccm_weil_eigenpair", b"test state"),
            cutoff: "13".into(),
            modes: n,
            precision: p,
            coefficients: vec![Float::with_val(p, 1); d],
            eigenvalue: "1".into(),
            selection_policy: None,
        };
        let a = (0..d * d)
            .map(|k| Float::with_val(p, u32::from(k / d == k % d)))
            .collect::<Vec<_>>();
        let m = RetainedMatrix::from_admitted_runtime(
            manifest("ccm_tau_matrix", b"identity matrix"),
            s.cutoff.clone(),
            n,
            p,
            &a,
        )
        .unwrap();
        let i = prepare_arithmetic_inputs(&s, &m).unwrap();
        let f = i.run_once.as_ref().unwrap().tail_form.as_ref().unwrap();
        for k in 0..f.dimension * f.dimension {
            let actual = scalar(&f.finite_zero_form[k], p).unwrap()
                + scalar(&f.tail_correction[k], p).unwrap();
            let expected = scalar(&f.lattice_gram[k], p).unwrap() / 2u32;
            assert!(
                (actual - &expected).abs()
                    < (expected.abs() + 1u32) * (Float::with_val(p, 1) >> 200)
            );
        }
        let lattice = i
            .atoms
            .iter()
            .filter(|a| a.family == "lattice")
            .collect::<Vec<_>>();
        assert_eq!(lattice.len(), n + 1);
        assert_eq!(scalar(&lattice[0].coordinate, p).unwrap(), 0);
        assert!(scalar(&lattice[0].weight, p).unwrap() > 0);
        let sum = lattice.iter().fold(Float::with_val(p, 0), |v, a| {
            v + scalar(&a.weight, p).unwrap()
        });
        assert!((sum - 1u32).abs() < (Float::with_val(p, 1) >> 200));
        let allowance = i.energy_allowance.as_ref().unwrap();
        assert!(scalar(&allowance.low_block_lower_bound, 384).unwrap() <= 1);
        assert!(scalar(&allowance.high_block_lower_bound, 384).unwrap() <= 1);
        assert!(scalar(&allowance.upper_trial_energy, 384).unwrap() >= 1);
        let mut o = ExtensionOptions::for_source(&s);
        o.working_precision_bits = i.precision_bits + 64;
        let result = weighted_tail_base(&s, &o, Some(&i)).unwrap();
        let origin = result
            .rows
            .iter()
            .find(|r| r.label.starts_with("lattice/"))
            .unwrap();
        assert!(!origin.values.contains_key("weighted_inverse_moment_1"));
        assert_eq!(origin.outcome, "unresolved_denominator");
        assert!(origin.notes.iter().any(|s| s.contains("origin")));
    }

    /// Opt-in local source replay; never runs in normal qualification or downloads data.
    #[test]
    fn interval_block_bound_recovers_positive_gap_when_gershgorin_fails() {
        let p = 192;
        let n = 3;
        let d = 7;
        let mut coefficients = vec![Float::with_val(p, 0); d];
        coefficients[n] = Float::with_val(p, 1);
        let s = RetainedState {
            manifest: manifest("ccm_weil_eigenpair", b"block bound test state"),
            cutoff: "13".into(),
            modes: n,
            precision: p,
            coefficients,
            eigenvalue: "0.001".into(),
            selection_policy: None,
        };
        let high = [0, 1, 5, 6];
        let a = (0..d * d)
            .map(|k| {
                let (i, j) = (k / d, k % d);
                scalar(
                    if i == n && j == n {
                        "0.001"
                    } else if i == j {
                        "1"
                    } else if high.contains(&i) && high.contains(&j) {
                        "0.4"
                    } else {
                        "0"
                    },
                    p,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let m = RetainedMatrix::from_admitted_runtime(
            manifest("ccm_tau_matrix", b"block matrix"),
            s.cutoff.clone(),
            n,
            p,
            &a,
        )
        .unwrap();
        let bounds = block_allowance(&s, &m, p + 128).unwrap();
        let mu = scalar(&bounds.high_block_lower_bound, p + 128).unwrap();
        assert!(mu > 0 && mu < scalar("0.6", p + 128).unwrap());
        assert!(mu > scalar(&bounds.upper_trial_energy, p + 128).unwrap());
        assert_eq!(scalar(&bounds.cross_block_norm_bound, p + 128).unwrap(), 0);
        assert!(bounds
            .hypotheses
            .iter()
            .any(|s| s.contains("all interval LDL pivots strictly positive")));
    }

    /// Opt-in local source replay; never runs in normal qualification or downloads data.
    #[test]
    #[ignore = "requires exact retained run bytes in XC_RESEARCH_PREPARATION_REPLAY"]
    fn retained_research_preparation_replay() -> Result<()> {
        use std::{fs, io::Write, path::PathBuf};
        let path = PathBuf::from(std::env::var("XC_RESEARCH_PREPARATION_REPLAY")?);
        let sm: ArtifactManifest =
            serde_json::from_slice(&fs::read(path.join("state.manifest.json"))?)?;
        let mm: ArtifactManifest =
            serde_json::from_slice(&fs::read(path.join("matrix.manifest.json"))?)?;
        let sb = fs::read(path.join("state.json"))?;
        let mb = fs::read(path.join("matrix.json"))?;
        ensure!(
            ContentDigest::sha256(&sb) == sm.content_digest
                && ContentDigest::sha256(&mb) == mm.content_digest,
            "source byte authentication failed"
        );
        let s = RetainedState::from_payload(&sm, &sb, std::slice::from_ref(&sm.content_digest))?;
        let value: serde_json::Value = serde_json::from_slice(&mb)?;
        ensure!(
            value["n_modes"] == s.modes
                && value["lambda_squared"] == s.cutoff
                && value["precision_bits"] == s.precision,
            "matrix source configuration differs"
        );
        let a = coeffs(
            &serde_json::from_value::<Vec<String>>(value["entries"].clone())?,
            s.precision,
        )?;
        // Same reader-admitted point matrix as the live run. Its exact parent
        // chain is separately checked by the exporter before these files exist.
        let m =
            RetainedMatrix::from_admitted_runtime(mm, s.cutoff.clone(), s.modes, s.precision, &a)?;
        let started = std::time::Instant::now();
        let i = prepare_arithmetic_inputs(&s, &m)?;
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path.join("prepared-inputs.json"))?;
        file.write_all(&serde_json::to_vec(&i)?)?;
        let mut o = ExtensionOptions::for_source(&s);
        o.working_precision_bits = i.precision_bits + 64;
        o.maximum_estimated_output_bytes = 8 << 30;
        let cache = ArtifactCacheContext {
            resolver: None,
            reference_resolver: None,
            acceptance: None,
            ordered_overlays: vec!["disabled".into()],
            mode: ArtifactExecutionCacheMode::Disabled,
            write_on_miss: false,
            write_visibility: CacheVisibility::Private,
            requested_assurance: xc_core::AssuranceLevel::Computed,
            certification_failure_policy: CertificationFailurePolicy::RetainComputedFailRun,
            production_sink: None,
        };
        let mut results = Vec::new();
        for id in [
            "weighted_tail",
            "energy_allowance",
            "tail_operator",
            "band_reconstruction",
        ] {
            let now = std::time::Instant::now();
            let result = capture_extended(
                id,
                &s,
                None,
                None,
                Some(&i),
                &o,
                std::slice::from_ref(&m.manifest),
                &cache,
            )?;
            results.push(serde_json::json!({"diagnostic":id,"seconds":now.elapsed().as_secs_f64(),"result":result.value}));
        }
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path.join("derived-replay.json"))?;
        file.write_all(&serde_json::to_vec(&serde_json::json!({"source":sm.content_digest,"seconds":started.elapsed().as_secs_f64(),"results":results}))?)?;
        Ok(())
    }
}
