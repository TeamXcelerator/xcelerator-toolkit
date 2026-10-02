use anyhow::{bail, Result};
use rug::{
    float::{Constant, Special},
    Float,
};

/// Natural logarithm of the leading prolate angular-deficiency predictor.
///
/// This evaluates log((2^14/3)*sqrt(2)*pi^5)-4*pi*c+(9/2)*log(c),
/// c=lambda^2. It is a point evaluation of a large-lambda asymptotic expression,
/// not a finite inequality or a Weil eigenvalue prediction theorem. c=1 is
/// accepted for algebraic evaluation only; c=0 and unsupported precision fail.
pub fn prolate_chi2_log_deficiency_asymptotic(c: u64, p: u32) -> Result<Float> {
    validate(c, p)?;
    let out = Float::with_val(p, log_at_precision(c, p + 64));
    if !out.is_finite() {
        bail!("asymptotic logarithm exceeds MPFR range");
    }
    Ok(out)
}

fn validate(c: u64, p: u32) -> Result<()> {
    if c == 0 || !(64..=1_000_000).contains(&p) {
        bail!("positive cutoff and 64..=1,000,000 precision required");
    }
    Ok(())
}

fn log_at_precision(c: u64, p: u32) -> Float {
    let pi = Float::with_val(p, Constant::Pi);
    let mut log_prefactor = Float::with_val(p, 2).ln();
    log_prefactor *= 29;
    log_prefactor /= 2;
    let mut term = pi.clone().ln();
    term *= 5;
    log_prefactor += term;
    log_prefactor -= Float::with_val(p, 3).ln();
    let mut log_power = Float::with_val(p, c).ln();
    log_power *= 9;
    log_power /= 2;
    let mut decay = pi;
    decay *= c;
    decay *= 4;
    log_prefactor += log_power;
    log_prefactor -= decay;
    log_prefactor
}

/// Checked positive point evaluation of the leading asymptotic expression.
/// If the positive value cannot be represented, use its logarithm instead.
pub fn try_prolate_chi2_deficiency_asymptotic(c: u64, p: u32) -> Result<Float> {
    validate(c, p)?;
    // Include the prefactor in the logarithm before exp. A separately
    // underflowed exponential cannot be rescued by multiplication afterward.
    let out = Float::with_val(p, log_at_precision(c, p + 64).exp());
    if !out.is_finite() || out <= 0 {
        bail!("positive asymptotic value is outside MPFR range; use log predictor");
    }
    Ok(out)
}

/// Compatibility wrapper for the positive asymptotic point expression.
/// Invalid precision/domain and unrepresentable positive values return NaN,
/// never an apparent exact zero. Use the checked or logarithmic APIs for errors.
pub fn prolate_chi2_deficiency_asymptotic(c: u64, p: u32) -> Float {
    try_prolate_chi2_deficiency_asymptotic(c, p)
        .unwrap_or_else(|_| Float::with_val(p.clamp(64, 1_000_000), Special::Nan))
}

/// Full PSWF index for an additive-even mode. This index is distinct from
/// both an even-sector concentration index and a Weil eigenvalue ordinal.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[doc(hidden)]
pub struct EvenProlateMode(u32);

impl EvenProlateMode {
    pub fn new(full_index: u32) -> Result<Self> {
        if !full_index.is_multiple_of(2) {
            bail!("an additive-even prolate mode requires an even full PSWF index");
        }
        Ok(Self(full_index))
    }

    pub fn full_index(self) -> u32 {
        self.0
    }

    /// Zero-based descending index in an additive-even concentration block.
    pub fn descending_even_index(self) -> u32 {
        self.0 / 2
    }

    /// Sign of the real finite-Fourier eigenvalue, not its singular value.
    pub fn fourier_sign(self) -> i32 {
        if self.0.is_multiple_of(4) {
            1
        } else {
            -1
        }
    }

    /// Convert to the ascending index expected by a selected-eigenvalue
    /// certificate for an additive-even concentration block of this size.
    /// The caller must establish that the matrix has this ordering and scope.
    pub fn ascending_even_concentration_index(self, dimension: usize) -> Result<usize> {
        let descending = usize::try_from(self.descending_even_index())?;
        dimension
            .checked_sub(descending + 1)
            .ok_or_else(|| anyhow::anyhow!("prolate mode is outside the even concentration block"))
    }
}

/// Natural log of the leading fixed-index predictor for `1 - |chi_n|`.
///
/// With c=lambda^2, this is log(C_n)+(n+1/2)log(c)-4*pi*c, where
/// C_n=2*sqrt(pi)*8^n*(2*pi)^(n+1/2)/n!. The positive singular-value
/// deficit is asymptotically half the concentration deficit `1-|chi_n|^2`.
/// For n=2 mod 4 the signed Fourier eigenvalue is negative; `1-chi_n`
/// is therefore NOT the small quantity computed here.
///
/// This is a point evaluation, not a finite-c enclosure or a uniform-in-n
/// asymptotic claim. Algebraic evaluation accepts c=1. n=4 preserves the
/// legacy arithmetic path exactly.
#[doc(hidden)]
pub fn prolate_log_deficiency_asymptotic(mode: EvenProlateMode, c: u64, p: u32) -> Result<Float> {
    validate(c, p)?;
    let out = Float::with_val(p, indexed_log_at_precision(mode, c, p + 64));
    if !out.is_finite() {
        bail!("asymptotic logarithm exceeds MPFR range");
    }
    Ok(out)
}

/// Positive point evaluation of the indexed leading expression. Refuses
/// overflow and nonzero-value underflow; use the log API for extreme cutoffs.
#[doc(hidden)]
pub fn try_prolate_deficiency_asymptotic(mode: EvenProlateMode, c: u64, p: u32) -> Result<Float> {
    validate(c, p)?;
    let out = Float::with_val(p, indexed_log_at_precision(mode, c, p + 64).exp());
    if !out.is_finite() || out <= 0 {
        bail!("positive asymptotic value is outside MPFR range; use log predictor");
    }
    Ok(out)
}

fn indexed_log_at_precision(mode: EvenProlateMode, c: u64, p: u32) -> Float {
    if mode.full_index() == 4 {
        return log_at_precision(c, p);
    }
    let n = u64::from(mode.full_index());
    let pi = Float::with_val(p, Constant::Pi);
    // log(C_n)=(4*n+3/2)*log(2)+(n+1)*log(pi)-log(n!).
    // Widen before integer arithmetic; no factorial or power is materialized.
    let mut result = Float::with_val(p, 2).ln();
    result *= 8 * n + 3;
    result /= 2;
    let mut term = pi.clone().ln();
    term *= n + 1;
    result += term;
    result -= Float::with_val(p, n + 1).ln_gamma();
    let mut power = Float::with_val(p, c).ln();
    power *= 2 * n + 1;
    power /= 2;
    result += power;
    let mut decay = pi;
    decay *= c;
    decay *= 4;
    result -= decay;
    result
}
