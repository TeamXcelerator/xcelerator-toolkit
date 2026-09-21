//! Deterministic atom contractions and bounded, authenticated band basis storage.
use super::{capture_runtime::Checkpoints, extended_research::coeffs};
use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use rug::{Assign, Float};
use serde::Serialize;
use std::{collections::VecDeque, path::PathBuf};
use xc_numerics::prefix::lossless_decimal as dec;

/// Fixed block boundaries and fixed final reduction order, independent of workers.
pub(crate) fn inner(a: &[Float], b: &[Float], w: &[Float], p: u32) -> Float {
    const BLOCK: usize = 512;
    let sums = a
        .par_chunks(BLOCK)
        .zip(b.par_chunks(BLOCK))
        .zip(w.par_chunks(BLOCK))
        .map(|((a, b), w)| {
            let mut sum = Float::with_val(p, 0);
            let mut term = Float::with_val(p, 0);
            for ((a, b), w) in a.iter().zip(b).zip(w) {
                term.assign(a);
                term *= b;
                term *= w;
                sum += &term;
            }
            sum
        })
        .collect::<Vec<_>>();
    sums.into_iter().fold(Float::with_val(p, 0), |s, x| s + x)
}
pub(crate) fn subtract(v: &mut [Float], q: &[Float], c: &Float, p: u32) {
    v.par_chunks_mut(512)
        .zip(q.par_chunks(512))
        .for_each(|(v, q)| {
            let mut term = Float::with_val(p, 0);
            for (v, q) in v.iter_mut().zip(q) {
                term.assign(c);
                term *= q;
                *v -= &term;
            }
        });
}
pub(crate) fn disk_budget() -> Result<u64> {
    let limit = std::env::var("XC_RESEARCH_BASIS_BYTES")
        .ok()
        .map(|v| v.parse::<u64>())
        .transpose()?
        .unwrap_or(8 << 30);
    if limit == 0 {
        bail!("XC_RESEARCH_BASIS_BYTES must be positive");
    }
    Ok(limit)
}
pub(crate) fn disk_estimate(rows: usize, p: u32, degree: usize) -> u64 {
    (rows as u64)
        .saturating_mul(degree as u64)
        .saturating_mul(u64::from(p).div_ceil(3) + 64)
        .saturating_add((degree as u64).saturating_mul(32768))
}
pub(crate) struct BandVectors {
    pub store: Checkpoints,
    cache: VecDeque<(usize, Vec<Float>)>,
    capacity: usize,
    rows: usize,
    p: u32,
    temporary: Option<PathBuf>,
}
impl BandVectors {
    pub fn new<T: Serialize>(
        identity: &T,
        rows: usize,
        p: u32,
        capacity: usize,
        degree: usize,
    ) -> Result<Self> {
        let store = Checkpoints::new(identity)?;
        let estimated = disk_estimate(rows, p, degree);
        let limit = disk_budget()?;
        if estimated > limit {
            bail!("band basis disk estimate {estimated} exceeds XC_RESEARCH_BASIS_BYTES={limit}");
        }
        let (store, temporary) = if store.enabled() {
            (store, None)
        } else {
            let path = std::env::temp_dir().join(format!(
                "xc-band-basis-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_nanos()
            ));
            std::fs::create_dir(&path)?;
            (Checkpoints::local(identity, path.clone())?, Some(path))
        };
        Ok(Self {
            store,
            cache: VecDeque::new(),
            capacity,
            rows,
            p,
            temporary,
        })
    }
    fn cache(&mut self, j: usize, v: Vec<Float>) {
        if self.capacity == 0 {
            return;
        }
        self.cache.retain(|(i, _)| *i != j);
        while self.cache.len() >= self.capacity {
            self.cache.pop_front();
        }
        self.cache.push_back((j, v));
    }
    pub fn save(&mut self, j: usize, v: Vec<Float>) -> Result<()> {
        if v.len() != self.rows {
            bail!("basis vector shape mismatch");
        }
        for (chunk, values) in v.chunks(4096).enumerate() {
            self.store.save(
                &format!("basis-{j}-{chunk}"),
                &values.iter().map(dec).collect::<Vec<_>>(),
            )?;
        }
        self.cache(j, v);
        Ok(())
    }
    pub fn get(&mut self, j: usize) -> Result<Vec<Float>> {
        if let Some((_, v)) = self.cache.iter().find(|(i, _)| *i == j) {
            return Ok(v.clone());
        }
        let mut result = Vec::with_capacity(self.rows);
        for start in (0..self.rows).step_by(4096) {
            let key = format!("basis-{j}-{}", start / 4096);
            let data = self
                .store
                .load::<Vec<String>>(&key)?
                .context("band basis block unavailable or corrupt")?;
            if data.len() != 4096.min(self.rows - start) {
                bail!("band basis block shape mismatch");
            }
            result.extend(coeffs(&data, self.p)?);
        }
        self.cache(j, result.clone());
        Ok(result)
    }
    pub fn clear_memory(&mut self) {
        self.cache.clear();
    }
}
impl Drop for BandVectors {
    fn drop(&mut self) {
        if let Some(path) = &self.temporary {
            let _ = std::fs::remove_dir_all(path);
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fixed_contractions_are_worker_independent_and_match_exact_sum() {
        let a = (0..2049)
            .map(|j| Float::with_val(128, j))
            .collect::<Vec<_>>();
        let one = vec![Float::with_val(128, 1); a.len()];
        let run = |workers| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap()
                .install(|| inner(&a, &one, &one, 128))
        };
        assert_eq!(run(1), run(4));
        assert_eq!(run(1), Float::with_val(128, 2048 * 2049 / 2));
    }
    #[test]
    fn basis_roundtrip_is_lossless_with_zero_memory_cache_and_multiple_blocks() {
        let values = (0..4100)
            .map(|j| Float::with_val(192, j) / 7u32)
            .collect::<Vec<_>>();
        let mut b = BandVectors::new(&"scratch-only-roundtrip", 4100, 192, 0, 2).unwrap();
        b.save(0, values.clone()).unwrap();
        assert_eq!(values, b.get(0).unwrap());
        b.save(1, values.clone()).unwrap();
        assert_eq!(values, b.get(1).unwrap());
    }
}
