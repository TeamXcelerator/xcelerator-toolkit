//! Exact dyadic finite polynomial forms, with explicit integer and workspace budgets.
//! A successful result is rounded once from the exact expression in stored points.
use anyhow::{bail, Result};
use rug::{Float, Integer};

const MAX_EXACT_BITS: u64 = 8_000_000;

#[derive(Clone)]
struct Dyadic {
    mantissa: Integer,
    exponent: i64,
}
impl Dyadic {
    fn zero() -> Self {
        Self {
            mantissa: Integer::new(),
            exponent: 0,
        }
    }
    fn from_float(x: &Float) -> Result<Self> {
        let (mantissa, exponent) = x
            .to_integer_exp()
            .ok_or_else(|| anyhow::anyhow!("nonfinite polynomial input"))?;
        let mut result = Self {
            mantissa,
            exponent: i64::from(exponent),
        };
        result.normalize();
        Ok(result)
    }
    fn normalize(&mut self) {
        if self.mantissa == 0 {
            self.exponent = 0;
            return;
        }
        let zeros = self.mantissa.find_one(0).unwrap();
        self.mantissa >>= zeros;
        self.exponent += i64::from(zeros);
    }
    fn bits(&self) -> u64 {
        u64::from(self.mantissa.significant_bits())
    }
    fn mul(&self, other: &Self, budget: &Budget, live: u64) -> Result<Self> {
        if self.mantissa == 0 || other.mantissa == 0 {
            return Ok(Self::zero());
        }
        let bits = self
            .bits()
            .checked_add(other.bits())
            .ok_or_else(|| anyhow::anyhow!("polynomial bit count overflow"))?;
        budget.check(bits, live)?;
        let exponent = self
            .exponent
            .checked_add(other.exponent)
            .ok_or_else(|| anyhow::anyhow!("polynomial exponent overflow"))?;
        let mut result = Self {
            mantissa: Integer::from(&self.mantissa * &other.mantissa),
            exponent,
        };
        result.normalize();
        Ok(result)
    }
    fn add(&self, other: &Self, budget: &Budget, live: u64) -> Result<Self> {
        if self.mantissa == 0 {
            return Ok(other.clone());
        }
        if other.mantissa == 0 {
            return Ok(self.clone());
        }
        let exponent = self.exponent.min(other.exponent);
        let left = u64::try_from(self.exponent - exponent)?;
        let right = u64::try_from(other.exponent - exponent)?;
        let bits = (self.bits() + left).max(other.bits() + right) + 1;
        budget.check(bits, live)?;
        let mut mantissa = self.mantissa.clone() << u32::try_from(left)?;
        mantissa += other.mantissa.clone() << u32::try_from(right)?;
        let mut result = Self { mantissa, exponent };
        result.normalize();
        Ok(result)
    }
    fn point(&self, p: u32) -> Result<Float> {
        let mut value = Float::with_val(p, &self.mantissa);
        if self.mantissa == 0 {
            return Ok(value);
        }
        let original = value.clone();
        let shift = u32::try_from(self.exponent.unsigned_abs())?;
        if self.exponent >= 0 {
            value <<= shift;
        } else {
            value >>= shift;
        }
        if !value.is_finite() || value.is_zero() {
            bail!("finite tail-form output exceeds exponent range");
        }
        let mut reverse = value.clone();
        if self.exponent >= 0 {
            reverse >>= shift;
        } else {
            reverse <<= shift;
        }
        if reverse != original {
            bail!("finite tail-form output scaling loses precision");
        }
        Ok(value)
    }
}
struct Budget {
    bytes: u64,
    fixed: u64,
}
impl Budget {
    fn check(&self, bits: u64, live: u64) -> Result<()> {
        // Eight temporary integer buffers conservatively cover shifted operands,
        // products, normalization and allocation growth. The live form storage is
        // included separately; this bounds calculation scratch, not host memory.
        if bits > MAX_EXACT_BITS {
            bail!("finite tail-form exact arithmetic exceeds eight-million-bit span budget");
        }
        let needed = bits
            .div_ceil(8)
            .saturating_mul(8)
            .saturating_add(live)
            .saturating_add(self.fixed);
        if needed > self.bytes {
            bail!("finite tail-form exact workspace budget exceeded");
        }
        Ok(())
    }
}
fn storage(values: &[Dyadic]) -> u64 {
    values.iter().map(|v| v.bits().div_ceil(8) + 64).sum()
}

pub(super) fn forms(
    basis: &[Vec<Float>],
    atoms: &[(Float, Float, bool)],
    p: u32,
    maximum_bytes: u64,
) -> Result<(Vec<Float>, Vec<Float>)> {
    let n = basis.len();
    if n == 0
        || n > 128
        || !(64..=1_000_000).contains(&p)
        || basis.iter().any(|v| {
            v.is_empty() || v.len() > 2049 || v.iter().any(|x| !x.is_finite() || x.prec() > p)
        })
        || atoms
            .iter()
            .any(|(x, w, _)| !x.is_finite() || !w.is_finite() || x.prec() > p || w.prec() > p)
    {
        bail!("invalid finite polynomial form shape or precision");
    }
    let budget = Budget {
        bytes: maximum_bytes,
        fixed: (basis.iter().map(Vec::len).sum::<usize>() as u64
            + atoms.len() as u64 * 2
            + n as u64 * n as u64 * 4
            + 64)
            .saturating_mul(u64::from(p).div_ceil(8) + 128)
            .saturating_mul(3),
    };
    budget.check(0, 0)?;
    let basis = basis
        .iter()
        .map(|v| v.iter().map(Dyadic::from_float).collect::<Result<Vec<_>>>())
        .collect::<Result<Vec<_>>>()?;
    let mut zero = vec![Dyadic::zero(); n * n];
    let mut lattice = zero.clone();
    for (x, w, is_zero) in atoms {
        let x = Dyadic::from_float(x)?;
        let w = Dyadic::from_float(w)?;
        if w.mantissa == 0 {
            continue;
        }
        let mut live = storage(&zero) + storage(&lattice);
        let mut values = Vec::with_capacity(n);
        for polynomial in &basis {
            let mut value = Dyadic::zero();
            for c in polynomial.iter().rev() {
                value = value.mul(&x, &budget, live)?.add(c, &budget, live)?;
            }
            live = live.saturating_add(value.bits().div_ceil(8) + 64);
            budget.check(value.bits(), live)?;
            values.push(value);
        }
        let target = if *is_zero { &mut zero } else { &mut lattice };
        for row in 0..n {
            for col in 0..=row {
                let value = w
                    .mul(&values[row], &budget, live)?
                    .mul(&values[col], &budget, live)?;
                let entry = target[row * n + col].add(&value, &budget, live)?;
                let copies = if row == col { 1 } else { 2 };
                live = live
                    .saturating_sub(copies * (target[row * n + col].bits().div_ceil(8) + 64))
                    .saturating_add(copies * (entry.bits().div_ceil(8) + 64));
                budget.check(entry.bits(), live)?;
                target[row * n + col] = entry.clone();
                target[col * n + row] = entry;
            }
        }
    }
    Ok((
        zero.iter().map(|v| v.point(p)).collect::<Result<_>>()?,
        lattice.iter().map(|v| v.point(p)).collect::<Result<_>>()?,
    ))
}
