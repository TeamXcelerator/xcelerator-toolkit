// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.)
// All rights reserved. See LICENSE in the repository root.

//! Optional exact CPU execution choices. None changes arithmetic or identity.

use anyhow::{bail, Result};
use rayon::prelude::*;

/// CPU residue-product implementation. The default preserves the portable path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CpuProductKernel {
    #[default]
    Portable,
    /// Contiguous signed-16-bit operands and a scalar exact dot product.
    Packed,
    /// Packed operands with AVX2; refuses if unavailable.
    Avx2,
    /// Packed operands with AVX-512F/BW; refuses if unavailable.
    Avx512,
    /// Runtime dispatch to AVX-512, AVX2, or packed scalar, in that order.
    Auto,
}

impl CpuProductKernel {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "portable" => Ok(Self::Portable),
            "packed" => Ok(Self::Packed),
            "avx2" => Ok(Self::Avx2),
            "avx512" => Ok(Self::Avx512),
            "auto" => Ok(Self::Auto),
            _ => bail!("XC_CPU_PRODUCT_KERNEL must be portable, packed, avx2, avx512, or auto"),
        }
    }

    /// Resolve optional ISA use against this process's available CPU features.
    pub fn resolve(self) -> Result<Self> {
        #[cfg(target_arch = "x86_64")]
        let features = (
            std::is_x86_feature_detected!("avx2"),
            std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw"),
        );
        #[cfg(not(target_arch = "x86_64"))]
        let features = (false, false);
        self.resolve_features(features.0, features.1)
    }

    fn resolve_features(self, avx2: bool, avx512: bool) -> Result<Self> {
        match self {
            Self::Avx2 if !avx2 => bail!("requested AVX2 CPU kernel is unavailable"),
            Self::Avx512 if !avx512 => bail!("requested AVX-512F/BW CPU kernel is unavailable"),
            Self::Auto => Ok(if avx512 {
                Self::Avx512
            } else if avx2 {
                Self::Avx2
            } else {
                Self::Packed
            }),
            value => Ok(value),
        }
    }
}

/// Storage of input residues. Flat storage removes one allocation per integer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ResidueLayout {
    #[default]
    Nested,
    Flat,
}

impl ResidueLayout {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "nested" => Ok(Self::Nested),
            "flat" => Ok(Self::Flat),
            _ => bail!("XC_RESIDUE_LAYOUT must be nested or flat"),
        }
    }
}

/// Execution-only settings, excluded from mathematical inputs and cache keys.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CpuExecutionOptions {
    pub kernel: CpuProductKernel,
    pub residue_layout: ResidueLayout,
}

impl CpuExecutionOptions {
    pub fn from_environment() -> Result<Self> {
        fn setting(name: &str, default: &str) -> Result<String> {
            match std::env::var(name) {
                Ok(value) => Ok(value),
                Err(std::env::VarError::NotPresent) => Ok(default.to_owned()),
                Err(error) => Err(error.into()),
            }
        }
        Ok(Self {
            kernel: CpuProductKernel::parse(&setting("XC_CPU_PRODUCT_KERNEL", "portable")?)?,
            residue_layout: ResidueLayout::parse(&setting("XC_RESIDUE_LAYOUT", "nested")?)?,
        })
    }
}

// Each centered residue has absolute value <= 8095. Every adjacent pair fits
// i32: 2 * 8095^2 < i32::MAX. Widen those pairs BEFORE accumulating. The caller's
// checked inner-dimension bound implies inner * 8095^2 < i64::MAX, bounding every
// SIMD partial sum and its horizontal reduction, not only the final answer.
pub(super) fn packed_product(
    a: &[u16],
    b: &[u16],
    m: usize,
    k: usize,
    n: usize,
    prime: u32,
    kernel: CpuProductKernel,
) -> Result<Vec<u16>> {
    let selected = kernel.resolve()?;
    let dot: unsafe fn(&[i16], &[i16]) -> i64 = match selected {
        CpuProductKernel::Packed => dot_scalar,
        #[cfg(target_arch = "x86_64")]
        CpuProductKernel::Avx2 => dot_avx2,
        #[cfg(target_arch = "x86_64")]
        CpuProductKernel::Avx512 => dot_avx512,
        _ => bail!("packed CPU kernel requires a supported packed implementation"),
    };
    let aa: Vec<i16> = a
        .iter()
        .map(|&v| super::centered(v, prime) as i16)
        .collect();
    let mut bt = vec![0i16; b.len()];
    bt.par_chunks_mut(k).enumerate().for_each(|(j, column)| {
        for (t, value) in column.iter_mut().enumerate() {
            *value = super::centered(b[t * n + j], prime) as i16;
        }
    });
    Ok((0..m * n)
        .into_par_iter()
        .map(|entry| {
            let (i, j) = (entry / n, entry % n);
            // The resolver checked every required ISA feature. Both slices
            // contain k initialized elements; SIMD loads include only full
            // vectors and the scalar tail retains all remaining elements.
            let sum = unsafe { dot(&aa[i * k..(i + 1) * k], &bt[j * k..(j + 1) * k]) };
            sum.rem_euclid(i64::from(prime)) as u16
        })
        .collect())
}

fn dot_scalar(a: &[i16], b: &[i16]) -> i64 {
    a.iter()
        .zip(b)
        .map(|(&a, &b)| i64::from(a) * i64::from(b))
        .sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_avx2(a: &[i16], b: &[i16]) -> i64 {
    use std::arch::x86_64::*;
    let mut low = _mm256_setzero_si256();
    let mut high = _mm256_setzero_si256();
    let end = a.len() / 16 * 16;
    for t in (0..end).step_by(16) {
        // Bounds and ISA requirements are established by packed_product.
        let pair = unsafe {
            _mm256_madd_epi16(
                _mm256_loadu_si256(a.as_ptr().add(t).cast()),
                _mm256_loadu_si256(b.as_ptr().add(t).cast()),
            )
        };
        low = _mm256_add_epi64(low, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(pair)));
        high = _mm256_add_epi64(
            high,
            _mm256_cvtepi32_epi64(_mm256_extracti128_si256::<1>(pair)),
        );
    }
    let mut lanes = [0i64; 4];
    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast(), _mm256_add_epi64(low, high)) };
    lanes.iter().sum::<i64>() + dot_scalar(&a[end..], &b[end..])
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn dot_avx512(a: &[i16], b: &[i16]) -> i64 {
    use std::arch::x86_64::*;
    let mut low = _mm512_setzero_si512();
    let mut high = _mm512_setzero_si512();
    let end = a.len() / 32 * 32;
    for t in (0..end).step_by(32) {
        let pair = unsafe {
            _mm512_madd_epi16(
                _mm512_loadu_si512(a.as_ptr().add(t).cast()),
                _mm512_loadu_si512(b.as_ptr().add(t).cast()),
            )
        };
        low = _mm512_add_epi64(low, _mm512_cvtepi32_epi64(_mm512_castsi512_si256(pair)));
        high = _mm512_add_epi64(
            high,
            _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64::<1>(pair)),
        );
    }
    let mut lanes = [0i64; 8];
    unsafe { _mm512_storeu_si512(lanes.as_mut_ptr().cast(), _mm512_add_epi64(low, high)) };
    lanes.iter().sum::<i64>() + dot_scalar(&a[end..], &b[end..])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optional_cpu_dispatch_defaults_and_unavailable_features_are_explicit() {
        assert_eq!(
            CpuExecutionOptions::default().kernel,
            CpuProductKernel::Portable
        );
        assert_eq!(
            CpuExecutionOptions::default().residue_layout,
            ResidueLayout::Nested
        );
        assert!(CpuProductKernel::parse("avx-512").is_err());
        assert!(ResidueLayout::parse("").is_err());
        assert!(CpuProductKernel::Avx512
            .resolve_features(true, false)
            .is_err());
        assert!(CpuProductKernel::Avx2
            .resolve_features(false, false)
            .is_err());
        assert_eq!(
            CpuProductKernel::Auto
                .resolve_features(false, false)
                .unwrap(),
            CpuProductKernel::Packed
        );
        assert_eq!(
            CpuProductKernel::Auto
                .resolve_features(true, false)
                .unwrap(),
            CpuProductKernel::Avx2
        );
        assert_eq!(
            CpuProductKernel::Auto.resolve_features(true, true).unwrap(),
            CpuProductKernel::Avx512
        );
    }

    #[test]
    fn packed_simd_signed_extremes_tails_and_unaligned_slices_match_i128_oracle() {
        for length in [
            1, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 511, 512, 513, 131_071,
        ] {
            // Prefix element deliberately misaligns the tested slices.
            let a: Vec<i16> = (0..length + 1)
                .map(|i| if i % 3 == 0 { -8095 } else { 8095 })
                .collect();
            let b: Vec<i16> = (0..length + 1)
                .map(|i| if i % 5 == 0 { 8095 } else { -8095 })
                .collect();
            let (a, b) = (&a[1..], &b[1..]);
            let expected: i128 = a
                .iter()
                .zip(b)
                .map(|(&x, &y)| i128::from(x) * i128::from(y))
                .sum();
            assert_eq!(i128::from(dot_scalar(a, b)), expected);
            #[cfg(target_arch = "x86_64")]
            {
                if CpuProductKernel::Avx2.resolve().is_ok() {
                    assert_eq!(i128::from(unsafe { dot_avx2(a, b) }), expected);
                }
                if CpuProductKernel::Avx512.resolve().is_ok() {
                    assert_eq!(i128::from(unsafe { dot_avx512(a, b) }), expected);
                }
            }
        }
    }
}
