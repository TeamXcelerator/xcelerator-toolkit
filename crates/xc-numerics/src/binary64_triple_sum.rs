//! Exact fixed-range sum of products of three finite binary64 stored values.
//! Three significands need at most 159 bits; their exponent range is finite.
//! Signed base-2^32 bins retain cancellation and round only the final total.
use anyhow::{bail, Result};
const BASE_EXP: i32 = -3328;
const BINS: usize = 208;
const RADIX: i128 = 1i128 << 32;
#[derive(Clone)]
pub(super) struct TripleSum {
    bins: [i128; BINS],
}
impl TripleSum {
    pub(super) fn new() -> Self {
        Self { bins: [0; BINS] }
    }
    pub(super) fn add(&mut self, values: [f64; 3]) -> Result<()> {
        if values.iter().any(|value| !value.is_finite()) {
            bail!("nonfinite product input");
        }
        let mut sign = 1i128;
        let mut exponent = 0i32;
        let mut product = vec![1u32];
        for value in values {
            if !value.is_finite() {
                bail!("nonfinite product input");
            }
            if value == 0.0 {
                return Ok(());
            }
            let bits = value.to_bits();
            if bits >> 63 != 0 {
                sign = -sign;
            }
            let encoded = ((bits >> 52) & 0x7ff) as i32;
            let significand =
                (bits & ((1u64 << 52) - 1)) | if encoded == 0 { 0 } else { 1u64 << 52 };
            exponent += if encoded == 0 {
                -1074
            } else {
                encoded - 1023 - 52
            };
            let factors = [significand as u32, (significand >> 32) as u32];
            let mut convolution = vec![0u128; product.len() + 2];
            for (i, &a) in product.iter().enumerate() {
                for (j, &b) in factors.iter().enumerate() {
                    convolution[i + j] += u128::from(a) * u128::from(b);
                }
            }
            for i in 0..convolution.len() - 1 {
                let carry = convolution[i] >> 32;
                convolution[i] &= u128::from(u32::MAX);
                convolution[i + 1] += carry;
            }
            product = convolution
                .into_iter()
                .map(|x| u32::try_from(x).expect("normalized short product limb"))
                .collect();
            while product.last() == Some(&0) {
                product.pop();
            }
        }
        let shift = usize::try_from(exponent - BASE_EXP)
            .map_err(|_| anyhow::anyhow!("binary64 product exponent below accumulator"))?;
        let offset = shift / 32;
        let bits = shift % 32;
        for (i, limb) in product.into_iter().enumerate() {
            let contribution = sign * (i128::from(limb) << bits);
            let cell = self
                .bins
                .get_mut(offset + i)
                .ok_or_else(|| anyhow::anyhow!("binary64 product exponent above accumulator"))?;
            *cell = cell
                .checked_add(contribution)
                .ok_or_else(|| anyhow::anyhow!("exact binary64 sum count overflow"))?;
        }
        Ok(())
    }
    pub(super) fn finish(self) -> Result<f64> {
        fn normalize(bins: &mut [i128; BINS]) -> Result<()> {
            for i in 0..BINS - 1 {
                let carry = bins[i].div_euclid(RADIX);
                bins[i] = bins[i].rem_euclid(RADIX);
                bins[i + 1] = bins[i + 1]
                    .checked_add(carry)
                    .ok_or_else(|| anyhow::anyhow!("binary64 accumulator carry overflow"))?;
            }
            Ok(())
        }
        let mut normalized = self.bins;
        normalize(&mut normalized)?;
        let negative = normalized[BINS - 1] < 0;
        if negative {
            for (out, input) in normalized.iter_mut().zip(self.bins) {
                *out = input
                    .checked_neg()
                    .ok_or_else(|| anyhow::anyhow!("binary64 accumulator sign overflow"))?;
            }
            normalize(&mut normalized)?;
        }
        if normalized[BINS - 1] >= RADIX {
            bail!("binary64 accumulator range exceeded");
        }
        let Some(top) = normalized.iter().rposition(|&x| x != 0) else {
            return Ok(0.0);
        };
        let mut high = top * 32 + (127 - normalized[top].leading_zeros() as usize);
        let mut exponent = BASE_EXP + high as i32;
        if exponent > 1023 {
            bail!("quadrature result overflows binary64");
        }
        let subnormal = exponent < -1022;
        let shift = if subnormal {
            (-1074 - BASE_EXP) as usize
        } else {
            high.saturating_sub(52)
        };
        let bit = |index: usize| -> bool {
            normalized
                .get(index / 32)
                .is_some_and(|&word| word & (1i128 << (index % 32)) != 0)
        };
        let mut significand = 0u64;
        if high >= shift {
            for index in (shift..=high).rev() {
                significand = (significand << 1) | u64::from(bit(index));
            }
        }
        if shift > 0 && bit(shift - 1) {
            let sticky = (0..shift - 1).any(bit);
            if sticky || significand & 1 != 0 {
                significand += 1;
            }
        }
        if significand == 0 {
            bail!("nonzero quadrature result underflows binary64");
        }
        if !subnormal && significand == (1u64 << 53) {
            significand >>= 1;
            high += 1;
            exponent = BASE_EXP + high as i32;
        }
        if exponent > 1023 {
            bail!("quadrature result overflows binary64");
        }
        let magnitude = if subnormal {
            significand
        } else {
            ((exponent + 1023) as u64) << 52 | (significand & ((1u64 << 52) - 1))
        };
        Ok(f64::from_bits((u64::from(negative) << 63) | magnitude))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn independently_generated_fraction_oracle_covers_ieee_boundaries_and_cancellation() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/data/triple_sum_fraction_oracle.json"
        ))
        .unwrap();
        for (index, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
            let mut sum = TripleSum::new();
            for term in case["terms"].as_array().unwrap() {
                let mut values = [0.0; 3];
                for (j, bits) in term.as_array().unwrap().iter().enumerate() {
                    values[j] =
                        f64::from_bits(u64::from_str_radix(bits.as_str().unwrap(), 16).unwrap());
                }
                sum.add(values).unwrap();
            }
            match case["expected_bits"].as_str() {
                Some(bits) => assert_eq!(
                    sum.finish().unwrap().to_bits(),
                    u64::from_str_radix(bits, 16).unwrap(),
                    "Fraction oracle {index}"
                ),
                None => assert!(
                    sum.finish().is_err(),
                    "Fraction oracle {index} expected range refusal"
                ),
            }
        }
        assert!(TripleSum::new().add([0.0, f64::NAN, 1.0]).is_err());
    }
    #[test]
    fn exact_products_retain_subnormal_terms_and_reveal_cancelled_small_components() {
        let tiny = f64::from_bits(1);
        let mut sum = TripleSum::new();
        sum.add([tiny, 2f64.powi(1023), 1.0]).unwrap();
        assert_eq!(sum.finish().unwrap(), 2f64.powi(-51));
        let mut sum = TripleSum::new();
        for term in [
            [2f64.powi(1000), 1.0, 1.0],
            [tiny, 1.0, 1.0],
            [-2f64.powi(1000), 1.0, 1.0],
        ] {
            sum.add(term).unwrap();
        }
        assert_eq!(sum.finish().unwrap(), tiny);
        let mut sum = TripleSum::new();
        sum.add([1.0 + f64::EPSILON, 1.0 - f64::EPSILON, 1.0])
            .unwrap();
        sum.add([-1.0, 1.0, 1.0]).unwrap();
        assert_eq!(sum.finish().unwrap(), -2f64.powi(-104));
    }
}
