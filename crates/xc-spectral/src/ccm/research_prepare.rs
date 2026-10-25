//! Derive finite research inputs from the retained run and the bundled ordinate table.
//! No result from another run, fitted energy, or assumed ordinate-tail constant is used.
use super::research_prepare_math as arithmetic;
use super::{
    convergence_capture::TailForm, extended_research::*, retained_evidence::*,
    state_geometry::RetainedState,
};
use anyhow::{ensure, Context, Result};
use rayon::prelude::*;
#[cfg(test)]
use rug::float::Constant;
use rug::Float;
use xc_cache::ContentDigest;
use xc_numerics::prefix::lossless_decimal as dec;

#[derive(Clone, Copy, Debug)]
struct PreparationShape {
    modes: usize,
    dimension: usize,
    degree: usize,
    prefix: usize,
    basis_dimension: usize,
    ordinates: usize,
}

impl PreparationShape {
    fn new(modes: usize, ordinates: usize) -> Result<Self> {
        ensure!(
            modes > 0,
            "arithmetic preparation requires a positive mode count"
        );
        ensure!(
            (1..=1_000_000).contains(&ordinates),
            "arithmetic preparation requires a nonempty admitted ordinate table"
        );
        let dimension = modes
            .checked_mul(2)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| anyhow::anyhow!("arithmetic preparation dimension overflow"))?;
        dimension
            .checked_mul(dimension)
            .context("arithmetic preparation matrix size overflow")?;
        modes
            .checked_mul(modes)
            .context("arithmetic preparation lattice size overflow")?;
        ensure!(
            u32::try_from(dimension).is_ok(),
            "arithmetic preparation factorial exponent exceeds its integer representation"
        );
        let degree = modes.min(64);
        let prefix = modes - degree;
        ensure!(
            prefix <= ordinates,
            "arithmetic preparation needs {prefix} prefix ordinates, but the bundled table has only {ordinates}; supply explicit inputs backed by sufficient ordinate data; no atoms were omitted"
        );
        Ok(Self {
            modes,
            dimension,
            degree,
            prefix,
            basis_dimension: degree + 1,
            ordinates,
        })
    }
}

fn preparation_float_bytes(precision: u32) -> u128 {
    u128::from(precision).div_ceil(64) * 8 + 96
}

fn preparation_output_bytes(shape: PreparationShape, precision: u32) -> Result<u64> {
    // Lossless decimal uses fewer than p/3 significant decimal digits.
    // Include strings, row metadata, promotion/validation, and serialization
    // scratch. This bounds this producer's shape, not arbitrary external JSON.
    let scalars = 2 * (shape.ordinates as u128 + shape.modes as u128 + 1)
        + 3 * (shape.basis_dimension as u128).pow(2)
        + 64;
    u64::try_from(4 * scalars * (u128::from(precision).div_ceil(3) + 256) + (1 << 20))
        .context("arithmetic preparation output estimate overflow")
}

fn preparation_working_bytes(
    shape: PreparationShape,
    source_precision: u32,
    precision: u32,
    workers: usize,
) -> Result<u64> {
    ensure!(
        workers > 0,
        "arithmetic preparation needs a positive worker count"
    );
    let dim = shape.dimension as u128;
    let side = shape.modes as u128 + 1;
    let basis = shape.basis_dimension as u128;
    let point = preparation_float_bytes(precision);
    let guarded = preparation_float_bytes(
        precision
            .checked_add(4096)
            .context("arithmetic preparation guard precision overflow")?,
    );
    // Block allowance: interval block, point copy, Householder copy, parallel
    // Schur columns, and linear scratch. Basis/row arrays remain resident.
    let blocks_and_basis =
        (12 * side * side + 8 * dim * basis + 64 * dim + 32 * 4096 + 2 * shape.ordinates as u128)
            * point;
    // Every simultaneously active transform worker owns a guarded coefficient
    // vector and bounded scalar/interval scratch; no table atom is dropped.
    let transforms = workers as u128 * (8 * dim + 512) * guarded;
    let state = dim * preparation_float_bytes(source_precision);
    let table_strings = shape.ordinates as u128 * 512;
    let output = u128::from(preparation_output_bytes(shape, precision)?);
    u64::try_from(blocks_and_basis + transforms + state + table_strings + output + (1 << 20))
        .context("arithmetic preparation workspace estimate overflow")
}

fn preparation_block_bytes(shape: PreparationShape, precision: u32) -> Result<u64> {
    let side = shape.modes as u128 + 1;
    u64::try_from(12 * side * side * preparation_float_bytes(precision))
        .context("arithmetic preparation block size overflow")
}

fn preparation_rational_bytes(
    shape: PreparationShape,
    coordinates: &[Float],
    length: &Float,
    workers: usize,
) -> Result<u64> {
    let point_bits = coordinates
        .iter()
        .chain(std::iter::once(length))
        .map(|value| {
            2 * u128::from(value.prec())
                + u128::from(i64::from(value.get_exp().unwrap_or(0)).unsigned_abs())
                + 8
        })
        .max()
        .unwrap_or(8);
    let n = shape.modes as u128;
    let prefix = shape.prefix as u128;
    let log = u128::from(usize::BITS - shape.dimension.leading_zeros());
    // Dyadic coordinate differences and rational lattice factors accumulate
    // exact numerator/denominator spans. Bound the complete products before
    // conversion, including the squared head coefficient. Each arithmetic
    // kernel independently enforces its per-value exact-rational bit budget.
    let basis_bits = 8 * n * log
        + prefix * (2 * point_bits + 4 * log + 8)
        + shape.basis_dimension as u128 * (4 * log + 8)
        + point_bits;
    let head_bits = 2 * (prefix * (2 * point_bits + 8) + n * (point_bits + 4 * log + 8)) + 128;
    let intermediate = basis_bits
        .max(head_bits)
        .min(2 * u128::from(arithmetic::RATIONAL_BIT_BUDGET));
    let worker_scratch = workers as u128 * (8 * intermediate.div_ceil(8) + 4096);
    let roots =
        (shape.ordinates as u128 + shape.prefix as u128 + 2) * (point_bits.div_ceil(8) + 96);
    let factorials = shape.dimension as u128 * ((2 * n * log).div_ceil(8) + 96);
    u64::try_from(worker_scratch + roots + factorials)
        .context("arithmetic preparation rational estimate overflow")
}

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
    match xc_core::RunResources::current() {
        Some(resources) => prepare_arithmetic_inputs_with_resources(state, matrix, &resources),
        None => {
            let resources = xc_core::RunResources::new(
                super::capture_runtime::CaptureResourcePolicy::from_environment()?
                    .maximum_working_bytes,
            )?;
            // Standalone borrowed parents have no shared run owner. Include
            // the already-resident matrix instead of treating it as free.
            let matrix_bytes = u64::try_from(
                matrix.entries.len() as u128 * preparation_float_bytes(matrix.precision) + 4096,
            )
            .context("arithmetic preparation parent estimate overflow")?;
            let _parent = resources
                .try_reserve_retained(matrix_bytes)
                .context("arithmetic preparation retained matrix exceeds working budget")?;
            prepare_arithmetic_inputs_with_resources(state, matrix, &resources)
        }
    }
}

/// Explicit shared-budget preparation for an already-admitted retained matrix.
///
/// The matrix must already be charged to `resources`; live borrowed matrices
/// and matrices decoded with `RetainedMatrix::from_reader_with_resources` meet
/// that contract. New work and the state view are reserved against remaining
/// capacity through validation and checkpoint serialization. The returned input
/// is owned by its caller, whose retained-output accounting remains separate.
pub fn prepare_arithmetic_inputs_with_resources(
    state: &RetainedState,
    matrix: &RetainedMatrix<'_>,
    resources: &xc_core::RunResources,
) -> Result<ExternalResearchInputs> {
    matrix.match_state(state)?;
    let _scope = resources.enter();
    let workers = arithmetic::preparation_workers()?;
    // The block eigensolver still uses the existing pool. Bounded producer
    // tasks must not mask that pool's independent worker admission.
    resources.check_workers(rayon::current_num_threads())?;
    let p = state
        .precision
        .checked_add(128)
        .context("arithmetic preparation precision overflow")?;
    precision(p)?;
    let table = xc_zeta::zeros::bundled_dataset_identity()?;
    let shape = PreparationShape::new(state.modes, table.record_count)?;
    let working = preparation_working_bytes(shape, state.precision, p, workers)?;
    let output = preparation_output_bytes(shape, p)?;
    let blocks = preparation_block_bytes(shape, p)?;
    // Dense interval/eigensolver blocks are not built until block_allowance.
    // Keep serialized output covered throughout, and reserve that later stage
    // only after the earlier Float arrays and exact coefficients are dropped.
    let arithmetic_working = working
        .checked_sub(output)
        .and_then(|bytes| bytes.checked_sub(blocks))
        .context("arithmetic preparation stage size overflow")?;
    let _output = resources.try_reserve_working(output).with_context(|| {
        format!("arithmetic preparation workspace requires {output} additional output bytes")
    })?;
    let arithmetic_workspace = resources
        .try_reserve_working(arithmetic_working)
        .with_context(|| {
            format!(
                "arithmetic preparation workspace requires {arithmetic_working} additional bytes"
            )
        })?;
    let _stage = super::capture_runtime::Stage::new("retained arithmetic research preparation");
    let definition = ContentDigest::sha256(&serde_json::to_vec(&(
        "retained-arithmetic-polynomial-v4",
        &matrix.manifest.content_digest,
        &table,
        state.modes,
        p,
    ))?);
    let store = super::capture_runtime::Checkpoints::new(&(
        "directed-point-stages-exact-tail-v3",
        &definition,
        &state.manifest.content_digest,
    ))?;
    if let Some(input) = store
        .limited_bytes(preparation_output_bytes(shape, p)? / 4)
        .load::<ExternalResearchInputs>("arithmetic-inputs")?
    {
        input.matches(state)?;
        return Ok(input);
    }
    let n = state.modes;
    let dim = shape.dimension;
    let l = finite_math::rounded_log_cutoff(&state.cutoff, p)?;
    let zeros = xc_zeta::zeros::bundled_first_n_strings(table.record_count)?;
    let ordinates = coeffs(&zeros, p)?;
    let z = arithmetic::coordinates(&ordinates, &l, n, p)?;
    // Fixed before any solve: retain at most degree 64 after a known-ordinate prefix.
    let degree = shape.degree;
    let prefix = shape.prefix;
    let d = shape.basis_dimension;
    let rational_bytes = preparation_rational_bytes(shape, &z, &l, workers)?;
    let scope = format!(
        "finite retained arithmetic section with correctly rounded point stages at {p} bits and exact dyadic head/tail splitting; coordinate z=(t/(2*pi*N/log(C)))^2; first {prefix} bundled ordinate decimals fixed as prefix; monomial basis 1,z,...,z^{degree} after that prefix; all {} bundled atoms through t={}; table sha256={}; arithmetic remainder=V^T*A*V/2-explicit head, includes all omitted signed form contributions of the stored matrix; no RH, complete critical-line head, continuum transfer, or source-accuracy premise",
        table.record_count,
        zeros.last().unwrap(),
        table.content_sha256
    );
    let mut input: ExternalResearchInputs = serde_json::from_value(serde_json::json!({
        "schema_version":1,"source_eigenpair":state.manifest.content_digest,
        "lambda_squared":state.cutoff,"n_modes":n,"precision_bits":p,
        "convention_id":"retained-arithmetic-polynomial-v4","definition_digest":definition,
        "approximation_scope":scope
    }))?;
    // State-weighted measures: transform squared and physical unit coefficient masses.
    input.atoms = arithmetic::preparation_map(ordinates.len(), workers, |j| -> Result<_> {
        let t = &ordinates[j];
        // Exact weight squaring reserves from its actual exponent span.
        // Workers must participate in the same run ledger as the owner.
        let _scope = resources.enter();
        let (v, _, _, _) = transform_terms(state, t, p)?;
        Ok(WeightedAtom {
            ordinal: j + 1,
            coordinate: dec(&z[j]),
            weight: dec(&arithmetic::square_sum(&[v], p)?),
            family: "zero".into(),
            partition: "bundled_decimal_head".into(),
        })
    })
    .into_iter()
    .collect::<Result<Vec<_>>>()?;
    let unit = source_unit(state, p)?;
    for k in 0..=n {
        let mass = arithmetic::square_sum(
            &if k == 0 {
                vec![unit[n].clone()]
            } else {
                vec![unit[n - k].clone(), unit[n + k].clone()]
            },
            p,
        )?;
        input.atoms.push(WeightedAtom {
            ordinal: k + 1,
            coordinate: dec(&arithmetic::round(&arithmetic::lattice(k, n), p)?),
            weight: dec(&mass),
            family: "lattice".into(),
            // Inverse moments divide by z, so the origin mode (z = 0) is its own
            // partition and the remaining modes keep finite inverse moments.
            partition: if k == 0 {
                "origin;ordinal=mode+1".into()
            } else {
                "nonzero_modes;ordinal=mode+1".into()
            },
        });
    }
    input.atom_coordinate = Some("z=(t/(2*pi*N/log(C)))^2; lattice z=(mode/N)^2".into());
    input.atom_coverage = Some(scope.clone());

    let v = {
        let _rational = resources.try_reserve_working(rational_bytes)
            .with_context(|| format!("arithmetic preparation exact coefficients require {rational_bytes} additional bytes"))?;
        arithmetic::basis(&z, &l, n, d, p)?
    };
    let (gram, actual) = arithmetic::forms(&matrix.entries, &v, dim, d, p)?;
    drop(v);
    let head = {
        let _rational = resources.try_reserve_working(rational_bytes)
            .with_context(|| format!("arithmetic preparation exact coefficients require {rational_bytes} additional bytes"))?;
        arithmetic::head(&ordinates, &z, &l, n, d, p)?
    };
    let (output_precision, tail, _promoted_output) =
        arithmetic::split(&actual, &head, p, resources, |target| {
            let promoted_bytes = preparation_output_bytes(shape, target)?
                .saturating_sub(preparation_output_bytes(shape, p)?);
            resources
                .try_reserve_working(promoted_bytes)
                .context("arithmetic preparation exact output exceeds remaining working budget")
        })?;
    let form=TailForm {definition_digest:definition.clone(),dimension:d,finite_zero_form:head.iter().map(dec).collect(),tail_correction:vec!["0".into();d*d],lattice_gram:gram.iter().map(dec).collect(),tail_operator_error:None,coverage:scope.clone(),hypotheses:vec!["Point arithmetic of the exact retained parent; assembly/source error is not bounded here. Model energy is computed without retained eigenvalue or eigenvector inputs.".into()],polynomial_coordinate:Some(input.atom_coordinate.clone().unwrap())};
    input.run_once = Some(super::convergence_capture::RunOnceInputs {
        tail_form: Some(form),
        ..Default::default()
    });
    let tail_correction = tail.iter().map(dec).collect();
    drop((actual, gram, head, tail, l, z, ordinates, zeros, unit));
    drop(arithmetic_workspace);
    // Reuse the original conservative stage estimate rather than assuming
    // that dropping reservations alone freed still-owned native storage.
    let _blocks = resources
        .try_reserve_working(working - output)
        .context("arithmetic preparation block bounds exceed remaining workspace")?;
    input.energy_allowance = Some(block_allowance(state, matrix, p)?);
    super::research_completion::promote_input_precision(&mut input, output_precision)?;
    input
        .run_once
        .as_mut()
        .unwrap()
        .tail_form
        .as_mut()
        .unwrap()
        .tail_correction = tail_correction;
    input.validate()?;
    if let Err(e) = store.save("arithmetic-inputs", &input) {
        xc_core::progress_message!("arithmetic preparation checkpoint unavailable: {e}");
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
/// `start + term(0) + ... + term(count - 1)`, added serially in index order;
/// the terms are formed on workers in bounded chunks.
fn ordered_sum(
    start: xc_numerics::mpfr_interval::MpfrInterval,
    count: usize,
    workers: usize,
    term: impl Fn(usize) -> xc_numerics::mpfr_interval::MpfrInterval + Sync,
) -> xc_numerics::mpfr_interval::MpfrInterval {
    const CHUNK: usize = 4096;
    let mut sum = start;
    for first in (0..count).step_by(CHUNK) {
        // Keep the original bounded term buffer. Limit simultaneous tasks,
        // rather than introducing one parallel barrier per worker-sized slice.
        let terms =
            arithmetic::preparation_map(count.min(first + CHUNK) - first, workers, |offset| {
                term(first + offset)
            });
        for value in terms {
            sum = sum.add(&value);
        }
    }
    sum
}
fn block_allowance(s: &RetainedState, m: &RetainedMatrix<'_>, p: u32) -> Result<EnergyAllowance> {
    use xc_numerics::mpfr_interval::MpfrInterval as I;
    let _stage = super::capture_runtime::Stage::new("finite matrix block bounds");
    let workers = arithmetic::preparation_workers()?;
    // Interval terms are formed on workers; every sum and elimination keeps
    // its serial order, so each bound is the serial bound bit for bit.
    xc_numerics::mpfr_interval::ensure_uniform_exponent_range()?;
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
        let mut block = arithmetic::preparation_map(size * size, workers, |t| {
            a(indices[t / size], indices[t % size])
        });
        let gersh = arithmetic::preparation_map(size, workers, |i| {
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
        .into_iter()
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
                    // Column i of the Schur update reads only column j, which
                    // this step leaves unchanged; columns update independently.
                    for first in (j + 1..size).step_by(workers) {
                        let columns = (first..size.min(first + workers))
                            .into_par_iter()
                            .map(|i| -> Result<Vec<I>> {
                                let factor = block[i * size + j].div(&pivot)?;
                                Ok((i..size)
                                    .map(|k| {
                                        block[k * size + i].sub(&block[k * size + j].mul(&factor))
                                    })
                                    .collect())
                            })
                            .collect::<Vec<_>>();
                        for (i, column) in (first..size).zip(columns) {
                            for (k, value) in (i..size).zip(column?) {
                                block[k * size + i] = value;
                            }
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
    let cross = ordered_sum(I::from_i64(0, p), low.len() * high.len(), workers, |t| {
        a(low[t / high.len()], high[t % high.len()]).square()
    });
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
    let energy = ordered_sum(I::from_i64(0, p), d * d, workers, |t| {
        a(t / d, t % d).mul(&v[t / d]).mul(&v[t % d])
    });
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
    #[test]
    fn ordered_interval_sum_keeps_large_batches_ordered_and_workers_bounded() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use xc_numerics::mpfr_interval::MpfrInterval as I;
        let count = 8193;
        let term = |index: usize| {
            I::point(Float::with_val(64, (index % 31) as i64 - 15) >> ((index * 17) % 193) as u32)
        };
        let expected = (0..count).fold(I::from_i64(0, 64), |sum, index| sum.add(&term(index)));
        let active = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        let calls = AtomicUsize::new(0);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();
        let actual = pool.install(|| {
            ordered_sum(I::from_i64(0, 64), count, 3, |index| {
                let live = active.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(live, Ordering::SeqCst);
                calls.fetch_add(1, Ordering::SeqCst);
                std::thread::yield_now();
                let value = term(index);
                active.fetch_sub(1, Ordering::SeqCst);
                value
            })
        });
        assert_eq!(calls.load(Ordering::SeqCst), count);
        assert_eq!(active.load(Ordering::SeqCst), 0);
        assert!((1..=3).contains(&peak.load(Ordering::SeqCst)));
        assert_eq!(actual.lower().as_ord(), expected.lower().as_ord());
        assert_eq!(actual.upper().as_ord(), expected.upper().as_ord());
    }

    #[test]
    fn arithmetic_keeps_worker_admission_for_the_unthrottled_eigensolver() {
        let p = 64;
        let state = RetainedState {
            manifest: manifest("ccm_weil_eigenpair", b"worker admission state"),
            cutoff: "13".into(),
            modes: 1,
            precision: p,
            coefficients: [0, 1, 0].map(|value| Float::with_val(p, value)).to_vec(),
            eigenvalue: "1".into(),
            selection_policy: None,
        };
        let entries = [1, 0, 0, 0, 1, 0, 0, 0, 1].map(|value| Float::with_val(p, value));
        let matrix = RetainedMatrix::from_admitted_runtime(
            manifest("ccm_tau_matrix", b"worker admission matrix"),
            state.cutoff.clone(),
            1,
            p,
            &entries,
        )
        .unwrap();
        let resources = xc_core::RunResources::with_worker_limit(1 << 20, 1).unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let error = pool
            .install(|| prepare_arithmetic_inputs_with_resources(&state, &matrix, &resources))
            .unwrap_err();
        assert!(error.to_string().contains("worker"));
        assert_eq!(resources.usage().working_bytes, 0);
        assert!(xc_core::RunResources::current().is_none());
    }

    #[test]
    fn arithmetic_stage_reservations_do_not_hold_future_dense_blocks() {
        for (modes, source_precision) in [(120, 3386), (800, 6708), (970, 6708)] {
            let shape = PreparationShape::new(modes, 1000).unwrap();
            let p = source_precision + 128;
            let working = preparation_working_bytes(shape, source_precision, p, 16).unwrap();
            let output = preparation_output_bytes(shape, p).unwrap();
            let blocks = preparation_block_bytes(shape, p).unwrap();
            assert!(blocks > 0 && working > blocks + output);
            let early = working - blocks;
            let resources = xc_core::RunResources::new(working).unwrap();
            let output_owner = resources.try_reserve_working(output).unwrap();
            let early_owner = resources.try_reserve_working(early - output).unwrap();
            assert_eq!(resources.usage().working_bytes, early);
            drop(early_owner);
            let late_owner = resources.try_reserve_working(working - output).unwrap();
            assert_eq!(resources.usage().working_bytes, working);
            drop((late_owner, output_owner));
            assert_eq!(resources.usage().working_bytes, 0);
        }
    }

    #[test]
    fn arithmetic_admission_uses_resources_and_keeps_the_full_ordinate_prefix() {
        let table = xc_zeta::zeros::bundled_dataset_identity().unwrap();
        let shape = PreparationShape::new(800, table.record_count).unwrap();
        assert_eq!(
            (shape.dimension, shape.prefix, shape.basis_dimension),
            (1601, 736, 65)
        );
        assert_eq!(shape.ordinates, table.record_count);
        assert!(PreparationShape::new(table.record_count + 64, table.record_count).is_ok());
        let error = PreparationShape::new(table.record_count + 65, table.record_count).unwrap_err();
        assert!(error.to_string().contains("no atoms were omitted"));
        assert!(PreparationShape::new(usize::MAX, table.record_count).is_err());

        let source_precision = 3386;
        let p = source_precision + 128;
        let working = preparation_working_bytes(shape, source_precision, p, 128).unwrap();
        let coordinates = [Float::with_val(p, 1), Float::with_val(p, 4)];
        let rational =
            preparation_rational_bytes(shape, &coordinates, &Float::with_val(p, 3), 128).unwrap();
        let matrix = u64::try_from(
            (shape.dimension as u128).pow(2) * preparation_float_bytes(source_precision) + 4096,
        )
        .unwrap();
        // A conservative exponent-span fixture for the separate exact-form
        // reservation; no N=800 matrix or numerical calculation is allocated.
        let exact_form = u64::try_from(
            ((shape.dimension as u128).pow(2) * u128::from(source_precision + 8192 + 2)
                + shape.dimension as u128
                    * shape.basis_dimension as u128
                    * u128::from(p + 16_384 + 2))
                * (shape.basis_dimension as u128 + 4)
                / 8,
        )
        .unwrap();
        let resources = xc_core::RunResources::with_worker_limit(512 << 30, 128).unwrap();
        let _matrix = resources.try_reserve_retained(matrix).unwrap();
        let _working = resources.try_reserve_working(working).unwrap();
        let _rational = resources.try_reserve_working(rational).unwrap();
        let _exact_form = resources.try_reserve_working(exact_form).unwrap();
        assert!(resources.usage().peak_bytes < 512 << 30);
        let small = xc_core::RunResources::new(working - 1).unwrap();
        assert!(small.try_reserve_working(working).is_err());
        assert_eq!(small.usage().working_bytes, 0);
    }

    #[test]
    fn arithmetic_low_remaining_budget_refuses_before_stage_allocation() {
        let p = 64;
        let state = RetainedState {
            manifest: manifest("ccm_weil_eigenpair", b"bounded arithmetic state"),
            cutoff: "13".into(),
            modes: 1,
            precision: p,
            coefficients: [0, 1, 0].map(|value| Float::with_val(p, value)).to_vec(),
            eigenvalue: "1".into(),
            selection_policy: None,
        };
        let entries = [1, 0, 0, 0, 1, 0, 0, 0, 1].map(|value| Float::with_val(p, value));
        let matrix = RetainedMatrix::from_admitted_runtime(
            manifest("ccm_tau_matrix", b"bounded arithmetic matrix"),
            state.cutoff.clone(),
            1,
            p,
            &entries,
        )
        .unwrap();
        let resources = xc_core::RunResources::new(1 << 20).unwrap();
        let retained = resources.try_reserve_retained((1 << 20) - 1).unwrap();
        let error =
            prepare_arithmetic_inputs_with_resources(&state, &matrix, &resources).unwrap_err();
        assert!(error.to_string().contains("workspace requires"));
        assert_eq!(resources.usage().working_bytes, 0);
        assert_eq!(resources.usage().retained_bytes, retained.bytes());
        assert!(xc_core::RunResources::current().is_none());
    }

    #[test]
    fn parallel_block_allowance_is_bit_identical_to_serial_reference_at_any_thread_count() {
        let p = 256;
        let mut paths = std::collections::BTreeSet::new();
        for (n, coupling) in [(4, "0.3"), (6, "0.05"), (5, "1.5")] {
            let d = 2 * n + 1;
            let coefficients = (0..d)
                .map(|k| Float::with_val(p, (k * 37 % 11) as i64 - 5) >> 3)
                .collect::<Vec<_>>();
            let s = RetainedState {
                manifest: manifest("ccm_weil_eigenpair", b"parallel block bound state"),
                cutoff: "13".into(),
                modes: n,
                precision: p,
                coefficients,
                eigenvalue: "0.001".into(),
                selection_policy: None,
            };
            let coupling = scalar(coupling, p).unwrap();
            let a = (0..d * d)
                .map(|k| {
                    let (i, j) = (k / d, k % d);
                    if i == j {
                        Float::with_val(p, 1 + (i % 3) as i64)
                    } else {
                        Float::with_val(p, &coupling / (1 + i.abs_diff(j) + (k % 5)) as u32)
                    }
                })
                .collect::<Vec<_>>();
            let m = RetainedMatrix::from_admitted_runtime(
                manifest("ccm_tau_matrix", b"parallel block matrix"),
                s.cutoff.clone(),
                n,
                p,
                &a,
            )
            .unwrap();
            let expected = reference_block_allowance(&s, &m, p + 128).unwrap();
            for note in &expected.hypotheses {
                for path in [
                    "interval LDL pivots strictly positive",
                    "outward Gershgorin",
                ] {
                    if note.contains(path) {
                        paths.insert(path);
                    }
                }
            }
            for threads in [1, 2, 4, 8] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                for _ in 0..2 {
                    let actual = pool.install(|| block_allowance(&s, &m, p + 128)).unwrap();
                    assert_eq!(actual, expected, "n={n} at {threads} threads");
                }
            }
        }
        // Both the interval LDL and the Gershgorin fallback were exercised.
        assert_eq!(paths.len(), 2);
    }

    /// Serial `block_allowance` retained verbatim from before parallel terms.
    fn reference_block_allowance(
        s: &RetainedState,
        m: &RetainedMatrix<'_>,
        p: u32,
    ) -> Result<EnergyAllowance> {
        use xc_numerics::mpfr_interval::MpfrInterval as I;
        let _stage = crate::ccm::capture_runtime::Stage::new("finite matrix block bounds");
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

    use xc_cache::*;
    fn exhaustive_preparation_fixture() -> ExternalResearchInputs {
        let p = 64;
        let n = 3;
        let d = 7;
        let state = RetainedState {
            manifest: manifest("ccm_weil_eigenpair", b"exact preparation closure"),
            cutoff: "13".into(),
            modes: n,
            precision: p,
            coefficients: vec![Float::with_val(p, 1); d],
            eigenvalue: "1".into(),
            selection_policy: None,
        };
        let values = (0..d * d)
            .map(|k| Float::with_val(p, u32::from(k / d == k % d)))
            .collect::<Vec<_>>();
        let matrix = RetainedMatrix::from_admitted_runtime(
            manifest("ccm_tau_matrix", b"identity preparation closure"),
            state.cutoff.clone(),
            n,
            p,
            &values,
        )
        .unwrap();
        prepare_arithmetic_inputs(&state, &matrix).unwrap()
    }
    #[test]
    fn exhaustive_preparation_lattice_coordinates_round_once() {
        let input = exhaustive_preparation_fixture();
        let point = input
            .atoms
            .iter()
            .find(|a| a.family == "lattice" && a.ordinal == 2)
            .unwrap();
        let p = input.precision_bits;
        assert_eq!(
            scalar(&point.coordinate, p).unwrap(),
            Float::with_val(p, Float::with_val(192, rug::Rational::from((1, 9))))
        );
    }
    #[test]
    fn exhaustive_preparation_ordinate_coordinate_rounds_once() {
        let input = exhaustive_preparation_fixture();
        let p = 64 + 128;
        let l = finite_math::rounded_log_cutoff("13", p).unwrap();
        let text = xc_zeta::zeros::bundled_first_n_strings(1).unwrap();
        let t = scalar(&text[0], p).unwrap();
        let want =
            (Float::with_val(2048, t) * l / (Float::with_val(2048, Constant::Pi) * 6u32)).square();
        let got = scalar(&input.atoms[0].coordinate, input.precision_bits).unwrap();
        assert_eq!(Float::with_val(p, got), Float::with_val(p, want));
    }
    #[test]
    fn exhaustive_preparation_tail_split_preserves_exact_identity_form() {
        let input = exhaustive_preparation_fixture();
        let p = input.precision_bits;
        let form = input.run_once.unwrap().tail_form.unwrap();
        for ((h, t), g) in form
            .finite_zero_form
            .iter()
            .zip(&form.tail_correction)
            .zip(&form.lattice_gram)
        {
            let h = scalar(h, p).unwrap().to_rational().unwrap();
            let t = scalar(t, p).unwrap().to_rational().unwrap();
            let g = scalar(g, p).unwrap().to_rational().unwrap();
            assert_eq!(h + t, g / 2);
        }
    }

    #[test]
    fn exhaustive_research_log_preparation_uses_exact_decimal_length() {
        let p = 64;
        let work = p + 128;
        let state = RetainedState {
            manifest: manifest("ccm_weil_eigenpair", b"exact-log-preparation-state"),
            cutoff: "1.000001".into(),
            modes: 1,
            precision: p,
            coefficients: [0, 1, 0].map(|x| Float::with_val(p, x)).to_vec(),
            eigenvalue: "1".into(),
            selection_policy: None,
        };
        let matrix = [1, 0, 0, 0, 1, 0, 0, 0, 1].map(|x| Float::with_val(p, x));
        let retained = RetainedMatrix::from_admitted_runtime(
            manifest("ccm_tau_matrix", b"exact-log-preparation-matrix"),
            state.cutoff.clone(),
            1,
            p,
            &matrix,
        )
        .unwrap();
        let actual = prepare_arithmetic_inputs(&state, &retained).unwrap();
        let exact = rug::Rational::from((1_000_001, 1_000_000));
        let mut lo = Float::with_val_round(2048, &exact, rug::float::Round::Down).0;
        let mut hi = Float::with_val_round(2048, &exact, rug::float::Round::Up).0;
        lo.ln_round(rug::float::Round::Down);
        hi.ln_round(rug::float::Round::Up);
        let length = Float::with_val(work, lo);
        assert_eq!(length, Float::with_val(work, hi));
        assert_eq!(actual.convention_id, "retained-arithmetic-polynomial-v4");

        let first = xc_zeta::zeros::bundled_first_n_strings(1).unwrap();
        let ordinate = scalar(&first[0], work).unwrap();
        let expected = Float::with_val(
            work,
            (Float::with_val(2048, ordinate) * length
                / (Float::with_val(2048, Constant::Pi) * 2u32))
                .square(),
        );
        assert_eq!(scalar(&actual.atoms[0].coordinate, work).unwrap(), expected);
    }
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
            "created_unix_seconds":1,"producer_toolkit_version":ToolkitVersion::parse("0.16.0").unwrap(),
            "minimum_reader_version":ToolkitVersion::parse("0.16.0").unwrap(),"maximum_reader_version":null,
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
        // The origin mode is its own partition: its mass is reported without
        // an inverse moment, and the nonzero modes keep finite inverse moments.
        let origin = result
            .rows
            .iter()
            .find(|r| r.label.starts_with("lattice/origin"))
            .unwrap();
        assert!(origin.values.contains_key("included_mass"));
        assert!(!origin.values.contains_key("weighted_inverse_moment_1"));
        assert_eq!(origin.outcome, "point_measurement");
        assert!(origin.notes.iter().any(|s| s.contains("origin")));
        let nonzero = result
            .rows
            .iter()
            .filter(|r| r.label.starts_with("lattice/nonzero_modes"))
            .collect::<Vec<_>>();
        assert!(!nonzero.is_empty());
        assert!(nonzero.iter().all(|r| r.outcome == "point_measurement"
            && r.values.contains_key("weighted_inverse_moment_1")));
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
