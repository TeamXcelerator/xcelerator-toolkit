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
