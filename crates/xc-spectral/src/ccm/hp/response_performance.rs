//! Reuse fixed root geometry without changing the rounded response formula.
use super::*;

// Avoid retaining an unbounded K-by-(2N+1) MPFR table. Derivatives are still
// prepared once when the optional denominator table exceeds this budget.
const GEOMETRY_BUDGET_BYTES: u128 = 512 * 1024 * 1024;
const ROOT_BATCH_SIZE: usize = 16;

struct PreparedRoot<'a> {
    value: &'a Float,
    denominators: Option<Vec<Float>>,
    derivative: Float,
}

pub(super) struct PreparedRootResponses<'a> {
    roots: Vec<Option<PreparedRoot<'a>>>,
    poles: &'a [Float],
    precision_bits: u32,
}

// Same adjacent-pair tree as deterministic_pairwise_sum_hp; leaf allocations
// survive across roots in a batch. No reassociation or reciprocal multiplication.
struct RootScratch {
    leaves: Vec<Float>,
    denominator: Float,
}
impl RootScratch {
    fn new(dimension: usize, bits: u32) -> Self {
        Self {
            leaves: (0..dimension).map(|_| Float::with_val(bits, 0)).collect(),
            denominator: Float::with_val(bits, 0),
        }
    }
    fn sum(&mut self, bits: u32) -> Float {
        let mut count = self.leaves.len();
        if count == 0 {
            return Float::with_val(bits, 0);
        }
        while count > 1 {
            let (left, right) = self.leaves.split_at_mut(1);
            left[0] += &right[0];
            for index in 1..count.div_ceil(2) {
                let (destination, source) = self.leaves.split_at_mut(2 * index);
                destination[index].assign(&source[0]);
                if 2 * index + 1 < count {
                    destination[index] += &source[1];
                }
            }
            count = count.div_ceil(2);
        }
        self.leaves[0].clone()
    }
}

impl<'a> PreparedRootResponses<'a> {
    pub(super) fn new(
        unit: &[Float],
        poles: &'a [Float],
        roots: &'a [EigenvalueResult],
        bits: u32,
    ) -> Result<Self> {
        Self::with_budget(unit, poles, roots, bits, GEOMETRY_BUDGET_BYTES)
    }

    fn with_budget(
        unit: &[Float],
        poles: &'a [Float],
        roots: &'a [EigenvalueResult],
        bits: u32,
        budget: u128,
    ) -> Result<Self> {
        if unit.len() != poles.len() {
            bail!("prime-power root response received incompatible source dimensions");
        }
        let bytes = (roots.len() as u128)
            .saturating_mul(poles.len() as u128)
            .saturating_mul(u128::from(bits).div_ceil(8) + 64);
        let retain_denominators = bytes <= budget;
        let prepared = roots
            .par_iter()
            .map(|outcome| {
                outcome
                    .value()
                    .map(|value| {
                        let mut derivative_terms = Vec::with_capacity(unit.len());
                        let mut denominators =
                            retain_denominators.then(|| Vec::with_capacity(unit.len()));
                        for (weight, pole) in unit.iter().zip(poles) {
                            let mut denominator = Float::with_val(bits, value);
                            denominator -= pole;
                            if denominator.is_zero() {
                                bail!("prime-power root response encountered a secular pole");
                            }
                            if let Some(stored) = &mut denominators {
                                stored.push(denominator.clone());
                            }
                            denominator.square_mut();
                            let mut derivative = Float::with_val(bits, weight);
                            derivative /= denominator;
                            derivative = -derivative;
                            derivative_terms.push(derivative);
                        }
                        let derivative = xc_numerics::reduction::deterministic_pairwise_sum_hp(
                            &derivative_terms,
                            bits,
                        );
                        if derivative.is_zero() {
                            bail!("prime-power root response has a zero secular derivative");
                        }
                        Ok(PreparedRoot {
                            value,
                            denominators,
                            derivative,
                        })
                    })
                    .transpose()
            })
            .collect::<Vec<Result<_>>>()
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            roots: prepared,
            poles,
            precision_bits: bits,
        })
    }

    pub(super) fn values(&self, tangent: &[Float]) -> Result<Vec<Option<String>>> {
        if tangent.len() != self.poles.len() {
            bail!("prime-power root response received incompatible source dimensions");
        }
        // Fixed chunks bound scratch use, retain source order and avoid a
        // separate MPFR allocation tree for every root of every event.
        let chunks = self
            .roots
            .par_chunks(ROOT_BATCH_SIZE)
            .map(|roots| {
                let mut scratch = RootScratch::new(tangent.len(), self.precision_bits);
                roots
                    .iter()
                    .map(|root| {
                        root.as_ref().map(|root| {
                            for (index, leaf) in scratch.leaves.iter_mut().enumerate() {
                                leaf.assign(&tangent[index]);
                                if let Some(denominators) = &root.denominators {
                                    *leaf /= &denominators[index];
                                } else {
                                    scratch.denominator.assign(root.value);
                                    scratch.denominator -= &self.poles[index];
                                    *leaf /= &scratch.denominator;
                                }
                            }
                            let mut response = scratch.sum(self.precision_bits);
                            response = -response;
                            response /= &root.derivative;
                            lossless_hp_decimal(&response)
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        Ok(chunks.into_iter().flatten().collect())
    }
}

// Bound concurrent event workspaces while allowing each event's root batches
// and matrix rows to use the shared Rayon pool. Indexed collection fixes output
// order and the caller reports the first failure in canonical event order.
pub(super) fn event_chunk_size(events: usize) -> usize {
    events
        .div_ceil(rayon::current_num_threads().clamp(1, 16))
        .max(1)
}

pub(super) struct ResponseProgress {
    phase: &'static str,
    total: usize,
    started: Instant,
    state: std::sync::Mutex<(Instant, usize)>,
}
impl ResponseProgress {
    pub(super) fn new(phase: &'static str, total: usize, roots: usize) -> Self {
        let started = Instant::now();
        if total >= 64 {
            eprintln!("[HP] prime-power response {phase}: 0/{total} events; {roots} roots/event; {} workers", rayon::current_num_threads());
        }
        Self {
            phase,
            total,
            started,
            state: std::sync::Mutex::new((started, 0)),
        }
    }
    pub(super) fn completed(&self) {
        let mut state = self.state.lock().expect("response progress mutex poisoned");
        state.1 += 1;
        if self.total >= 64 && (state.1 == self.total || state.0.elapsed().as_secs() >= 30) {
            eprintln!(
                "[HP] prime-power response {}: {}/{} events; elapsed {:.1}s",
                self.phase,
                state.1,
                self.total,
                self.started.elapsed().as_secs_f64()
            );
            state.0 = Instant::now();
        }
    }
}

/// A process-local seal for the exact freshly computed response. Its producer
/// already ran spectral-isolation and bordered-residual gates. Cache reads never
/// receive this seal and must replay the numerical validator. Stream JSON so an
/// additional multi-gigabyte copy is never needed just to bind this witness.
#[derive(Default)]
pub(super) struct FreshResponseSeal(RefCell<Option<ContentDigest>>);
impl FreshResponseSeal {
    fn digest(value: &impl Serialize) -> std::result::Result<ContentDigest, CacheError> {
        use sha2::{Digest, Sha256};
        use std::io::{BufWriter, Write};
        struct HashWriter(Sha256);
        impl Write for HashWriter {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0.update(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let mut output = BufWriter::with_capacity(256 * 1024, HashWriter(Sha256::new()));
        serde_json::to_writer(&mut output, value)
            .map_err(|error| CacheError::Serialization(error.to_string()))?;
        output
            .flush()
            .map_err(|error| CacheError::Serialization(error.to_string()))?;
        let hash = output
            .into_inner()
            .map_err(|error| CacheError::Serialization(error.to_string()))?;
        Ok(ContentDigest(format!("{:x}", hash.0.finalize())))
    }
    pub(super) fn record(&self, value: &impl Serialize) -> std::result::Result<(), CacheError> {
        self.0.replace(Some(Self::digest(value)?));
        Ok(())
    }
    pub(super) fn verify_fresh(
        &self,
        value: &impl Serialize,
    ) -> std::result::Result<bool, CacheError> {
        let stored = self.0.borrow();
        let Some(expected) = stored.as_ref() else {
            return Ok(false);
        };
        if Self::digest(value)? != *expected {
            return Err(CacheError::InvalidManifest(
                "fresh response changed after its numerical production gates".into(),
            ));
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn outcome(value: Float) -> EigenvalueResult {
        let bits = value.prec();
        EigenvalueResult::Converged(RootRefinement {
            value,
            diagnostics: RootRefinementDiagnostics {
                iterations: 1,
                final_correction: Float::with_val(bits, 0),
                residual: Float::with_val(bits, 0),
                achieved_decimal_digits: Float::with_val(bits, 20),
            },
        })
    }
    fn source(
        bits: u32,
        n: usize,
        count: usize,
    ) -> (Vec<Float>, Vec<Float>, Vec<EigenvalueResult>, Vec<Float>) {
        let poles = (-(n as i64)..=n as i64)
            .map(|i| Float::with_val(bits, i))
            .collect::<Vec<_>>();
        let unit = poles
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let mut value = Float::with_val(bits, i + 1);
                value /= 2 * n + 3;
                value
            })
            .collect::<Vec<_>>();
        let tangent = unit
            .iter()
            .enumerate()
            .map(|(i, x)| if i % 2 == 0 { x.clone() } else { -x.clone() })
            .collect();
        let roots = (0..count)
            .map(|i| {
                let mut x = Float::with_val(bits, i + 1);
                x /= 8;
                x += n + 1;
                outcome(x)
            })
            .collect();
        (unit, poles, roots, tangent)
    }
    fn reference(
        unit: &[Float],
        poles: &[Float],
        roots: &[EigenvalueResult],
        tangent: &[Float],
        bits: u32,
    ) -> Vec<Option<String>> {
        roots
            .iter()
            .map(|root| {
                root.value().map(|value| {
                    lossless_hp_decimal(
                        &prime_power_root_velocity_response(unit, tangent, poles, value, bits)
                            .unwrap(),
                    )
                })
            })
            .collect()
    }
    #[test]
    fn prepared_root_responses_preserve_bits_threads_and_budget_fallback() {
        for bits in [128, 192, 1024, 3386, 6708] {
            let (unit, poles, mut roots, tangent) = source(bits, 8, 35);
            let mut near_pole = Float::with_val(bits, 2).pow(-((bits / 3) as i32));
            near_pole += 2;
            roots.push(outcome(near_pole));
            let mut negative = Float::with_val(bits, -17);
            negative /= 3;
            roots.push(outcome(negative));
            roots.insert(
                17,
                EigenvalueResult::Failed {
                    iterations: 0,
                    reason: "retained failure".into(),
                },
            );
            let expected = reference(&unit, &poles, &roots, &tangent, bits);
            for workers in [1, 2, 4] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(workers)
                    .build()
                    .unwrap();
                for budget in [0, GEOMETRY_BUDGET_BYTES] {
                    pool.install(|| {
                        let prepared =
                            PreparedRootResponses::with_budget(&unit, &poles, &roots, bits, budget)
                                .unwrap();
                        assert_eq!(prepared.values(&tangent).unwrap(), expected);
                        let zero = vec![Float::with_val(bits, 0); unit.len()];
                        assert_eq!(
                            prepared.values(&zero).unwrap(),
                            reference(&unit, &poles, &roots, &zero, bits)
                        );
                    });
                }
            }
        }
    }
    #[test]
    fn prepared_root_responses_preserve_invalid_geometry_rejection() {
        let (unit, poles, roots, tangent) = source(192, 2, 1);
        let at_pole = vec![outcome(poles[0].clone())];
        assert!(PreparedRootResponses::new(&unit, &poles, &at_pole, 192)
            .err()
            .unwrap()
            .to_string()
            .contains("secular pole"));
        let zero = vec![Float::with_val(192, 0); unit.len()];
        assert!(PreparedRootResponses::new(&zero, &poles, &roots, 192).is_err());
        assert!(PreparedRootResponses::new(&unit[..2], &poles, &roots, 192).is_err());
        let prepared = PreparedRootResponses::new(&unit, &poles, &roots, 192).unwrap();
        assert!(prepared.values(&tangent[..2]).is_err());
    }
    #[test]
    fn fresh_seal_is_exact_and_never_available_for_cached_or_altered_data() {
        let value = serde_json::json!({"source":"A", "roots":["0", "-0", "1e-2000"], "nested":{"response":"9"}});
        let seal = FreshResponseSeal::default();
        assert!(!seal.verify_fresh(&value).unwrap());
        assert_eq!(
            FreshResponseSeal::digest(&value).unwrap(),
            ContentDigest::sha256(&serde_json::to_vec(&value).unwrap())
        );
        seal.record(&value).unwrap();
        assert!(seal.verify_fresh(&value).unwrap());
        for key in ["source", "roots", "nested"] {
            let mut changed = value.clone();
            changed[key] = serde_json::Value::Null;
            assert!(seal.verify_fresh(&changed).is_err());
        }
        assert!(!FreshResponseSeal::default().verify_fresh(&value).unwrap());
    }
    #[test]
    #[ignore = "explicit bounded release-mode response-kernel benchmark"]
    fn response_root_kernel_benchmark() {
        let bits = 6708;
        let (unit, poles, roots, tangent) = source(bits, 400, 400);
        let repeats = 3;
        let started = Instant::now();
        let expected = (0..repeats)
            .map(|_| reference(&unit, &poles, &roots, &tangent, bits))
            .collect::<Vec<_>>();
        let baseline_seconds = started.elapsed().as_secs_f64();
        let expected_digest = ContentDigest::sha256(&serde_json::to_vec(&expected).unwrap());
        for workers in [1, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap();
            pool.install(|| {
                let preparation = Instant::now();
                let prepared = PreparedRootResponses::new(&unit, &poles, &roots, bits).unwrap();
                let preparation_seconds = preparation.elapsed().as_secs_f64();
                let started = Instant::now();
                let actual = (0..repeats).map(|_| prepared.values(&tangent).unwrap()).collect::<Vec<_>>();
                let candidate_seconds = started.elapsed().as_secs_f64();
                assert_eq!(actual, expected);
                eprintln!("RESPONSE_BENCH {}", serde_json::json!({"bits":bits,"dimension":801,"roots":400,"events":repeats,"workers":workers,"baseline_seconds":baseline_seconds,"preparation_seconds":preparation_seconds,"candidate_seconds":candidate_seconds,"payload_sha256":expected_digest.0,"bit_identical":true}));
            });
        }
    }
}
