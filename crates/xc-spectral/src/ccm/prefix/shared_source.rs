//! Canonical large-source handoff for the retained even-sector reader.
//! Shares an already charged and validated runtime matrix; never rebuilds the
//! whole JSON payload or changes its identity. Construction validation remains
//! the responsibility of the ordinary cache resolver before this handoff.
use super::*;
use serde::ser::{Error as _, SerializeSeq};
use sha2::{Digest, Sha256};
use std::io::Write;
use xc_cache::SharedValidated;
use xc_core::RunResources;

// Rug 1.30 Display reserves an intermediate buffer using BOTH precision and
// exponent, even for scientific notation. Bound both before any to_string.
// With p <= 1,000,000 and |binary exponent| <= 100,001, decimal digits are
// <= ceil(p/3)+4. Rug's exponent reserve is < |binary exponent|+8. Four times
// their sum plus 128 bytes is < 2 MiB, covering its temporary/output strings
// and growth allowance. This is cooperative accounting, not an allocator/RSS
// theorem. The matrix itself keeps its existing allocation charge.
const SERIALIZATION_WORKING_BYTES: u64 = 2 << 20;
const MAXIMUM_FORMAT_POWER: i32 = 100_000;

fn check_formatting_domain(values: &[Float]) -> Result<()> {
    // Comparisons avoid Float::get_exp(), which can panic for exponents outside
    // i32 under an expanded MPFR range. These four boundary values use 64 bits.
    let upper = Float::with_val(64, 1) << MAXIMUM_FORMAT_POWER;
    let lower = Float::with_val(64, 1) >> MAXIMUM_FORMAT_POWER;
    if !upper.is_finite() || lower.is_zero() {
        bail!("shared retained source formatting requires a supported MPFR exponent range");
    }
    let negative_upper = -upper.clone();
    let negative_lower = -lower.clone();
    for value in values {
        if value > &upper
            || value < &negative_upper
            || (!value.is_zero() && value < &lower && value > &negative_lower)
        {
            bail!("shared retained source exponent exceeds the bounded formatting domain");
        }
    }
    Ok(())
}

struct Entries<'a> {
    values: &'a [Float],
    resources: &'a RunResources,
}
impl Serialize for Entries<'_> {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        let mut sequence = serializer.serialize_seq(Some(self.values.len()))?;
        for value in self.values {
            self.resources.check_cancelled().map_err(S::Error::custom)?;
            let text = value.to_string();
            if text.len() > 1_048_576 {
                return Err(S::Error::custom("prefix scalar exceeds the input budget"));
            }
            sequence.serialize_element(&text)?;
        }
        sequence.end()
    }
}

// Exact field order and scalar rendering of hp::PortableEvenSectorMatrix.
// Production-source parity tests bind this wire contract independently.
#[derive(Serialize)]
struct Payload<'a> {
    schema_version: u32,
    lambda_squared: &'a str,
    n_modes: usize,
    precision_bits: u32,
    dimension: usize,
    entries: Entries<'a>,
}

struct HashWriter<'a> {
    hash: Sha256,
    length: u64,
    maximum: u64,
    resources: &'a RunResources,
}
impl Write for HashWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.resources
            .check_cancelled()
            .map_err(std::io::Error::other)?;
        let next = self
            .length
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| std::io::Error::other("retained source payload length overflow"))?;
        if next > self.maximum {
            return Err(std::io::Error::other(
                "retained source exceeds its declared logical byte count",
            ));
        }
        self.hash.update(bytes);
        self.length = next;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.resources
            .check_cancelled()
            .map_err(std::io::Error::other)
    }
}

impl RetainedEvenMatrix {
    /// Internal handoff after the ordinary resolver has validated this exact
    /// source against Tau. Logical length may exceed the legacy byte decoder's
    /// cap; only bounded formatting scratch and the live shared matrix are used.
    /// Noncanonical source spellings are refused rather than silently relabeled.
    pub(in crate::ccm) fn from_shared_source(
        manifest: &ArtifactManifest,
        cutoff: &str,
        modes: usize,
        p: u32,
        entries: SharedValidated<Vec<Float>>,
        allowed: &[ContentDigest],
        resources: &RunResources,
    ) -> Result<Self> {
        resources.check_cancelled()?;
        manifest.validate()?;
        if manifest.quality.admissible_rank() < CacheQuality::Validated.admissible_rank()
            || !manifest.immutable
            || !allowed.contains(&manifest.content_digest)
            || manifest.size_bytes > resources.maximum_bytes()
        {
            bail!("source must match an explicitly approved immutable payload within the run byte budget");
        }
        if manifest.key.kind != "ccm_even_sector_matrix" {
            bail!("an ordered even-sector matrix is required");
        }
        precision(p)?;
        let dimension = modes
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("invalid retained even-matrix shape or metadata"))?;
        if dimension > 8193
            || dimension.checked_mul(dimension) != Some(entries.len())
            || cutoff.trim().is_empty()
        {
            bail!("invalid retained even-matrix shape or metadata");
        }
        validate_cutoff(cutoff)?;
        let expected_retained_bytes = (entries.len() as u64)
            .checked_mul(64 + u64::from(p).div_ceil(64) * 8)
            .and_then(|bytes| bytes.checked_add(64))
            .ok_or_else(|| anyhow::anyhow!("retained source runtime size overflow"))?;
        if !entries.is_charged_to(resources) || entries.retained_bytes() < expected_retained_bytes {
            bail!("shared retained source must keep its complete allocation charge on the same run ledger");
        }
        if entries
            .iter()
            .any(|value| !value.is_finite() || value.prec() != p)
        {
            bail!("shared retained matrix must contain finite scalars at its declared precision");
        }
        let _scratch = resources.try_reserve_working(SERIALIZATION_WORKING_BYTES)?;
        check_formatting_domain(&entries)?;
        let mut writer = HashWriter {
            hash: Sha256::new(),
            length: 0,
            maximum: manifest.size_bytes,
            resources,
        };
        // The payload contains only checked finite MPFR strings and integers;
        // serde_json uses the same bytes as the producer's finite_json encoder.
        let encoded = serde_json::to_writer(
            &mut writer,
            &Payload {
                schema_version: 1,
                lambda_squared: cutoff,
                n_modes: modes,
                precision_bits: p,
                dimension,
                entries: Entries {
                    values: &entries,
                    resources,
                },
            },
        );
        resources.check_cancelled()?;
        encoded?;
        if writer.length != manifest.size_bytes
            || format!("{:x}", writer.hash.finalize()) != manifest.content_digest.0
        {
            bail!("shared retained matrix does not replay its exact approved source digest and byte count");
        }
        for i in 0..dimension {
            resources.check_cancelled()?;
            for j in 0..i {
                if entries[i * dimension + j] != entries[j * dimension + i] {
                    bail!("source storage is not exactly symmetric");
                }
            }
        }
        resources.check_cancelled()?;
        Ok(Self {
            manifest: manifest.clone(),
            cutoff: cutoff.to_owned(),
            modes,
            precision: p,
            entries: MatrixEntries::Shared(entries),
        })
    }
}

#[cfg(test)]
mod tests;
