//! Private partial execution records. No portable artifact or numerical policy changes.
use super::*;
use crate::ccm::capture_runtime::{Checkpoints, Stage};
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicUsize, Ordering};

const BLOCK_MODES: usize = 32;

pub(super) struct Phase {
    name: String,
    started: std::time::Instant,
    complete: bool,
    _stage: Option<Stage>,
}
impl Phase {
    pub(super) fn new(name: impl Into<String>) -> Self {
        let name = name.into();
        Self {
            _stage: Some(Stage::new(name.clone())),
            name,
            started: std::time::Instant::now(),
            complete: false,
        }
    }
    pub(super) fn batch(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            started: std::time::Instant::now(),
            complete: false,
            // The enclosing prime stage already owns its heartbeat. Batches
            // need outcome accounting, not another temporary timer thread.
            _stage: None,
        }
    }
    pub(super) fn finish(mut self) {
        self.complete = true;
        xc_core::progress_message!(
            "u-flow phase {}: completed in {:.3}s",
            self.name,
            self.started.elapsed().as_secs_f64()
        );
    }
}
impl Drop for Phase {
    fn drop(&mut self) {
        if !self.complete {
            xc_core::progress_message!(
                "u-flow phase {}: incomplete (error, cancellation, or unwind) after {:.3}s",
                self.name,
                self.started.elapsed().as_secs_f64()
            );
        }
    }
}

pub(super) struct Partial<'a> {
    store: &'a Checkpoints,
    binding: String,
    budget: u128,
    resources: Option<xc_core::RunResources>,
    #[cfg(test)]
    stop_after_prime: usize,
}

// Hash exact stored dyadics without constructing enormous decimal strings or
// exponent-sized rationals. Scratch is one significand, reused for every point.
fn rule_digest(nodes: &[Float], weights: &[Float]) -> String {
    let mut hash = Sha256::new();
    let mut digits = Vec::new();
    for values in [nodes, weights] {
        hash.update((values.len() as u64).to_le_bytes());
        for value in values {
            hash.update(value.prec().to_le_bytes());
            hash.update([
                u8::from(value.is_sign_negative()),
                u8::from(value.is_finite()),
                u8::from(value.is_nan()),
            ]);
            hash.update(value.get_exp().unwrap_or(0).to_le_bytes());
            if let Some(integer) = value.get_significand() {
                let size = integer.significant_bits().div_ceil(8) as usize;
                digits.resize(size, 0);
                integer.write_digits(&mut digits, rug::integer::Order::Lsf);
                hash.update((size as u64).to_le_bytes());
                hash.update(&digits);
            } else {
                hash.update(0u64.to_le_bytes());
            }
        }
    }
    xc_core::hex_lower(&hash.finalize())
}

impl<'a> Partial<'a> {
    pub(super) fn new(
        store: Option<&'a Checkpoints>,
        orders: &[usize],
        tables: &std::collections::HashMap<usize, Table>,
        event_batch: usize,
        budget: u128,
    ) -> Option<Self> {
        let store = store.filter(|store| store.enabled())?;
        // Match the whole-action reuse gate even when no RunResources owner
        // installed a pool: heterogeneous MPFR worker ranges forbid reuse.
        xc_numerics::mpfr_interval::ensure_uniform_exponent_range().ok()?;
        // Unicode failure disables optional partial reuse; the original numerical
        // configuration validation still determines the original error.
        let policy = std::env::vars_os()
            .filter_map(|(name, value)| {
                let name = name.to_str()?;
                (name.starts_with("XC_PRIME_RESPONSE_")
                    || name.starts_with("XC_ROOT_RESPONSE_")
                    || name.starts_with("XC_CUDA_")
                    || name.starts_with("XC_DENSE_")
                    || name == "XC_RESPONSE_EVENT_BATCH")
                    .then(|| {
                        value
                            .into_string()
                            .ok()
                            .map(|value| (name.to_owned(), value))
                    })
            })
            .collect::<Option<BTreeMap<_, _>>>()?;
        // Table identity calculation is optional and admitted before allocation.
        let maximum_bits = tables
            .values()
            .flat_map(|(nodes, weights)| nodes.iter().chain(weights))
            .map(Float::prec)
            .max()
            .unwrap_or(0);
        let allowance = (tables.len() as u128 * 1024
            + orders.len() as u128 * 32
            + u128::from(maximum_bits).div_ceil(8) * 4
            + (1 << 20))
            .min(u128::from(u64::MAX));
        if allowance > budget {
            return None;
        }
        let _reservation = reserve(allowance as u64)?;
        let table_ids = tables
            .iter()
            .map(|(&order, (nodes, weights))| (order, rule_digest(nodes, weights)))
            .collect::<BTreeMap<_, _>>();
        let identity = serde_json::json!({"semantics": "uflow-partial-directed-v1", "orders": orders, "tables": table_ids, "event_batch": event_batch, "policies": policy, "mpfr_range": [rug::float::exp_min(), rug::float::exp_max()]});
        let bytes = serde_json::to_vec(&identity).ok()?;
        Some(Self {
            store,
            binding: ContentDigest::sha256(&bytes).0,
            budget,
            resources: xc_core::RunResources::current(),
            #[cfg(test)]
            stop_after_prime: STOP_AFTER_PRIME.with(std::cell::Cell::get),
        })
    }

    fn label(&self, kind: &str, work: u32, start: usize) -> String {
        format!("u-flow-partial-{}-{kind}-{work}-{start}", self.binding)
    }
    fn load(
        &self,
        kind: &str,
        work: u32,
        start: usize,
        count: usize,
        total: usize,
        step: usize,
    ) -> Option<(usize, Vec<I>)> {
        let cap = payload_cap(count, work)?;
        let allowance = (1u64 << 20).checked_add(cap.checked_mul(4)?)?;
        if u128::from(allowance) > self.budget {
            return None;
        }
        let _scope = self.resources.as_ref().map(xc_core::RunResources::enter);
        let _reservation = reserve(allowance)?;
        let saved = self
            .store
            .limited_bytes(cap)
            .load::<Intervals>(&self.label(kind, work, start))
            .ok()??;
        let decoded = saved.decode(work, start, count, total, step);
        match decoded {
            Ok(value) => Some(value),
            Err(error) => {
                xc_core::progress_message!(
                    "u-flow partial {kind} checkpoint rejected ({error}); recomputing"
                );
                None
            }
        }
    }
    fn save<'b>(
        &self,
        kind: &str,
        work: u32,
        start: usize,
        next: usize,
        count: usize,
        values: impl Iterator<Item = &'b I>,
    ) {
        let Some(cap) = payload_cap(count, work) else {
            return;
        };
        let Some(allowance) = cap
            .checked_mul(4)
            .and_then(|bytes| bytes.checked_add(1 << 20))
        else {
            return;
        };
        if u128::from(allowance) > self.budget {
            return;
        }
        let _scope = self.resources.as_ref().map(xc_core::RunResources::enter);
        let Some(_reservation) = reserve(allowance) else {
            return;
        };
        let saved = Intervals::encode(work, start, next, values);
        if let Err(error) = saved.and_then(|saved| {
            self.store
                .limited_bytes(cap)
                .save(&self.label(kind, work, start), &saved)
        }) {
            xc_core::progress_message!("u-flow partial {kind} checkpoint not saved ({error})");
        }
    }
    pub(super) fn prime_prefix(
        &self,
        work: u32,
        count: usize,
        total: usize,
        step: usize,
    ) -> Option<(usize, Vec<I>)> {
        self.load("prime-prefix", work, 0, count, total, step)
    }
    pub(super) fn save_prime(&self, work: u32, next: usize, values: &[I]) -> Result<()> {
        self.save("prime-prefix", work, 0, next, values.len(), values.iter());
        #[cfg(test)]
        {
            COMPUTED_PRIME_BATCHES.with(|value| value.set(value.get() + 1));
            if self.stop_after_prime != 0 && next >= self.stop_after_prime {
                bail!("injected interruption after completed u-flow prime prefix");
            }
        }
        Ok(())
    }
}

fn reserve(bytes: u64) -> Option<Option<xc_core::MemoryReservation>> {
    match xc_core::RunResources::current() {
        Some(resources) => resources.try_reserve_working(bytes).ok().map(Some),
        None => Some(None),
    }
}
fn payload_cap(count: usize, work: u32) -> Option<u64> {
    (count as u64)
        .checked_mul(2)?
        .checked_mul(u64::from(work) + 160)?
        .checked_add(2048)
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Intervals {
    schema: u32,
    work: u32,
    start: usize,
    next: usize,
    // Fixed number of strings prevents untrusted JSON Vec allocation amplification.
    endpoints: [String; 2],
}
impl Intervals {
    fn encode<'b>(
        work: u32,
        start: usize,
        next: usize,
        values: impl Iterator<Item = &'b I>,
    ) -> Result<Self> {
        let mut endpoints = [String::new(), String::new()];
        for value in values {
            value.validate()?;
            anyhow::ensure!(
                value.precision() == work,
                "partial interval precision differs"
            );
            for (encoded, value) in endpoints.iter_mut().zip([value.lower(), value.upper()]) {
                if !encoded.is_empty() {
                    encoded.push('\n');
                }
                // Canonical decimal retains signed zeros and every endpoint bit.
                encoded.push_str(&lossless_hp_decimal(value));
            }
        }
        Ok(Self {
            schema: 1,
            work,
            start,
            next,
            endpoints,
        })
    }
    fn decode(
        self,
        work: u32,
        start: usize,
        count: usize,
        total: usize,
        step: usize,
    ) -> Result<(usize, Vec<I>)> {
        anyhow::ensure!(
            self.schema == 1
                && self.work == work
                && self.start == start
                && self.next > start
                && self.next <= total
                && step > 0
                && (self.next == total || (self.next - start).is_multiple_of(step)),
            "partial interval prefix identity differs"
        );
        anyhow::ensure!(
            count > 0
                && self
                    .endpoints
                    .iter()
                    .all(|text| text.split('\n').count() == count
                        && text
                            .split('\n')
                            .all(|value| !value.is_empty() && value.len() <= work as usize + 128)),
            "partial interval shape differs"
        );
        let parse = |text: &str| -> Result<Float> {
            let value = parse_hp_scalar(text, work)?;
            anyhow::ensure!(
                value.is_finite() && lossless_hp_decimal(&value) == text,
                "partial interval is not canonical finite data"
            );
            Ok(value)
        };
        let values = self.endpoints[0]
            .split('\n')
            .zip(self.endpoints[1].split('\n'))
            .map(|(lo, hi)| Ok(I::new(parse(lo)?, parse(hi)?)?))
            .collect::<Result<Vec<_>>>()?;
        Ok((self.next, values))
    }
}

pub(super) fn integrals<S: Send, Init, F>(
    checkpoint: Option<&Partial<'_>>,
    work: u32,
    count: usize,
    init: Init,
    compute: F,
) -> Result<Vec<[I; 3]>>
where
    Init: Fn() -> S + Sync + Send,
    F: Fn(&mut S, usize) -> Result<[I; 3]> + Sync + Send,
{
    let Some(checkpoint) = checkpoint else {
        return (0..count).into_par_iter().map_init(init, compute).collect();
    };
    let mut result = Vec::with_capacity(count);
    // Only the contiguous authenticated prefix is reused. Files beyond the
    // first absent/corrupt block are ignored, even if parallel work saved them.
    for start in (0..count).step_by(BLOCK_MODES) {
        let end = (start + BLOCK_MODES).min(count);
        let Some((next, values)) = checkpoint.load(
            "integrals",
            work,
            start,
            (end - start) * 3,
            end,
            end - start,
        ) else {
            break;
        };
        if next != end {
            break;
        }
        let mut values = values.into_iter();
        for _ in start..end {
            result.push([
                values.next().unwrap(),
                values.next().unwrap(),
                values.next().unwrap(),
            ]);
        }
    }
    let reused = result.len();
    let completed = AtomicUsize::new(reused);
    xc_core::progress_message!(
        "u-flow integral blocks: reused {reused}/{count} modes at {work} bits"
    );
    let starts = (reused..count).step_by(BLOCK_MODES).collect::<Vec<_>>();
    let blocks = starts
        .into_par_iter()
        .map(|start| -> Result<Vec<[I; 3]>> {
            let end = (start + BLOCK_MODES).min(count);
            // Nested Rayon work retains all admitted workers even for few blocks.
            let values = (start..end)
                .into_par_iter()
                .map_init(&init, &compute)
                .collect::<Result<Vec<_>>>()?;
            checkpoint.save(
                "integrals",
                work,
                start,
                end,
                values.len() * 3,
                values.iter().flatten(),
            );
            let done = completed.fetch_add(end - start, Ordering::Relaxed) + end - start;
            xc_core::progress_message!(
                "u-flow integral blocks: completed {done}/{count} modes at {work} bits"
            );
            Ok(values)
        })
        .collect::<Result<Vec<_>>>()?;
    result.extend(blocks.into_iter().flatten());
    Ok(result)
}

#[cfg(test)]
thread_local! {
    pub(super) static STOP_AFTER_PRIME: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    pub(super) static COMPUTED_PRIME_BATCHES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(values: &[I]) -> Vec<(u32, String, bool, String, bool)> {
        values
            .iter()
            .map(|value| {
                (
                    value.precision(),
                    value.lower().to_string_radix(16, None),
                    value.lower().is_sign_negative(),
                    value.upper().to_string_radix(16, None),
                    value.upper().is_sign_negative(),
                )
            })
            .collect()
    }
    fn mode(index: usize, work: u32) -> [I; 3] {
        let point = Float::with_val(work, index + 1) / 7u32;
        [
            I::point(point.clone()),
            I::new(-point.clone(), point).unwrap(),
            I::new(
                Float::with_val(work, rug::float::Special::NegZero),
                Float::with_val(work, 0),
            )
            .unwrap(),
        ]
    }
    fn context(store: &Checkpoints) -> Partial<'_> {
        Partial {
            store,
            binding: "unit-test-context".into(),
            budget: 1 << 30,
            resources: xc_core::RunResources::current(),
            stop_after_prime: 0,
        }
    }

    #[test]
    fn uflow_partial_intervals_preserve_exact_endpoints_and_reject_malformed_records() {
        let values = mode(7, 192);
        let encoded = Intervals::encode(192, 0, 3, values.iter()).unwrap();
        let serialized = serde_json::to_vec(&encoded).unwrap();
        let decode = || serde_json::from_slice::<Intervals>(&serialized).unwrap();
        assert_eq!(
            bits(&decode().decode(192, 0, 3, 3, 3).unwrap().1),
            bits(&values)
        );
        assert!(decode().decode(193, 0, 3, 3, 3).is_err());
        assert!(decode().decode(192, 1, 3, 3, 3).is_err());
        assert!(decode().decode(192, 0, 2, 3, 3).is_err());
        let mut bad = decode();
        bad.next = 0;
        assert!(bad.decode(192, 0, 3, 3, 3).is_err());
        let mut bad = decode();
        bad.endpoints[0] = "NaN\n0\n0".into();
        assert!(bad.decode(192, 0, 3, 3, 3).is_err());
        let mut bad = decode();
        bad.endpoints[0].push_str("\n0");
        assert!(bad.decode(192, 0, 3, 3, 3).is_err());
    }

    #[test]
    fn uflow_partial_integrals_resume_contiguous_prefix_and_reject_corruption() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        pool.install(|| {
            let resources = xc_core::RunResources::new(1 << 30).unwrap();
            let _scope = resources.enter();
            let directory = xc_core::test_support::TestDir::new("uflow-partial-integrals");
            let store = Checkpoints::local(&"source A", directory.to_path_buf()).unwrap();
            let checkpoint = context(&store);
            let interrupted = integrals(
                Some(&checkpoint),
                192,
                70,
                || (),
                |_, index| {
                    if index == 40 {
                        bail!("injected mode interruption");
                    }
                    Ok(mode(index, 192))
                },
            );
            assert!(interrupted.is_err());
            let calls = AtomicUsize::new(0);
            let resumed = integrals(
                Some(&checkpoint),
                192,
                70,
                || (),
                |_, index| {
                    calls.fetch_add(1, Ordering::Relaxed);
                    Ok(mode(index, 192))
                },
            )
            .unwrap();
            assert_eq!(calls.load(Ordering::Relaxed), 38);
            let expected = (0..70)
                .flat_map(|index| mode(index, 192))
                .collect::<Vec<_>>();
            assert_eq!(
                bits(&resumed.into_iter().flatten().collect::<Vec<_>>()),
                bits(&expected)
            );
            let warm = integrals(
                Some(&checkpoint),
                192,
                70,
                || (),
                |_, _| -> Result<[I; 3]> { panic!("warm prefix recomputed") },
            )
            .unwrap();
            assert_eq!(
                bits(&warm.into_iter().flatten().collect::<Vec<_>>()),
                bits(&expected)
            );
            // Actual payload corruption must invalidate the first block and
            // ignore the otherwise valid later block files.
            let base = std::fs::read_dir(&directory)
                .unwrap()
                .next()
                .unwrap()
                .unwrap()
                .path();
            let label = checkpoint.label("integrals", 192, 0);
            let path = base.join(ContentDigest::sha256(label.as_bytes()).0);
            std::fs::write(path, b"corrupt").unwrap();
            calls.store(0, Ordering::Relaxed);
            let fresh = integrals(
                Some(&checkpoint),
                192,
                70,
                || (),
                |_, index| {
                    calls.fetch_add(1, Ordering::Relaxed);
                    Ok(mode(index, 192))
                },
            )
            .unwrap();
            assert_eq!(calls.load(Ordering::Relaxed), 70);
            assert_eq!(
                bits(&fresh.into_iter().flatten().collect::<Vec<_>>()),
                bits(&expected)
            );
            let changed = Checkpoints::local(&"source B", directory.to_path_buf()).unwrap();
            assert!(context(&changed)
                .load("integrals", 192, 0, 96, 32, 32)
                .is_none());
            let mut denied = context(&store);
            denied.budget = 0;
            assert!(denied.load("integrals", 192, 0, 96, 32, 32).is_none());
        });
    }

    #[test]
    fn uflow_partial_prime_snapshot_binds_batch_rule_and_endpoint_identity() {
        let directory = xc_core::test_support::TestDir::new("uflow-prime-prefix");
        let store = Checkpoints::local(&"source", directory.to_path_buf()).unwrap();
        let mut tables = std::collections::HashMap::from([(
            2,
            (
                vec![Float::with_val(128, -0.5), Float::with_val(128, 0.5)],
                vec![Float::with_val(128, 1); 2],
            ),
        )]);
        let checkpoint = Partial::new(Some(&store), &[2, 2], &tables, 3, 1 << 30).unwrap();
        let values = mode(5, 192);
        checkpoint.save_prime(192, 6, &values).unwrap();
        let (next, decoded) = checkpoint.prime_prefix(192, 3, 9, 3).unwrap();
        assert_eq!(next, 6);
        assert_eq!(bits(&decoded), bits(&values));
        assert!(checkpoint.prime_prefix(192, 3, 9, 4).is_none());
        assert!(checkpoint.prime_prefix(193, 3, 9, 3).is_none());
        let changed_batch = Partial::new(Some(&store), &[2, 2], &tables, 1, 1 << 30).unwrap();
        assert!(changed_batch.prime_prefix(192, 3, 9, 3).is_none());
        tables.get_mut(&2).unwrap().1[0] = Float::with_val(128, 2);
        let changed_table = Partial::new(Some(&store), &[2, 2], &tables, 3, 1 << 30).unwrap();
        assert!(changed_table.prime_prefix(192, 3, 9, 3).is_none());
        assert!(Partial::new(Some(&store), &[2, 2], &tables, 3, 0).is_none());
    }

    #[test]
    fn uflow_partial_reuse_disabled_for_mixed_worker_exponent_ranges() {
        unsafe extern "C" {
            fn mpfr_set_emin(value: std::os::raw::c_long) -> i32;
        }
        struct RestoreExponentMin(i32);
        impl Drop for RestoreExponentMin {
            fn drop(&mut self) {
                assert_eq!(unsafe { mpfr_set_emin(self.0 as _) }, 0);
            }
        }
        let normal = rug::float::exp_min();
        let directory = xc_core::test_support::TestDir::new("uflow-partial-worker-range");
        let store = Checkpoints::local(&"source", directory.to_path_buf()).unwrap();
        let tables = std::collections::HashMap::new();
        let uniform = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        uniform.install(|| {
            let checkpoint = Partial::new(Some(&store), &[], &tables, 3, 1 << 30).unwrap();
            checkpoint.save_prime(192, 3, &mode(5, 192)).unwrap();
            assert!(checkpoint.prime_prefix(192, 3, 6, 3).is_some());
        });
        let mixed = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .start_handler(|index| {
                if index != 0 {
                    assert_eq!(unsafe { mpfr_set_emin(-1000) }, 0);
                }
            })
            .build()
            .unwrap();
        mixed.install(|| {
            let _restore = RestoreExponentMin(rug::float::exp_min());
            assert_eq!(unsafe { mpfr_set_emin(normal as _) }, 0);
            assert!(xc_core::RunResources::current().is_none());
            assert!(xc_numerics::mpfr_interval::ensure_uniform_exponent_range().is_err());
            assert!(Partial::new(Some(&store), &[], &tables, 3, 1 << 30).is_none());
        });
    }
}
