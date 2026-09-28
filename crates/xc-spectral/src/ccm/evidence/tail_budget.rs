//! Outward-rounded evaluation of the finite archimedean tail budget.
use anyhow::{bail, Result};
use rug::{float::Round, Float};
use xc_numerics::{interval::RationalInterval, mpfr_interval::MpfrInterval as I};

/// An outward upper bound for the omitted archimedean tail in the stated
/// finite frequency space. Construction checks the theorem domain.
///
/// Mathematical fields are immutable; a label or a caller-supplied number
/// cannot replace the verified budget. This is not an eigenvalue certificate.
#[derive(Clone, Debug)]
pub struct ArchimedeanTailBudget {
    integer_cutoff_c: u64,
    log_cutoff: I,
    rho: I,
    modes: usize,
    cutoff_t: Float,
    theorem_threshold: Float,
    upper_bound: Float,
}

impl ArchimedeanTailBudget {
    /// Enclose the explicit expression in Groskin, Corollary 3.3(iii).
    ///
    /// Requires integer c > 1, N >= 1, finite T > max(2*pi*N/log(c),7),
    /// and 64 through 1,000,000 input precision bits. T is interpreted as
    /// its exact stored dyadic value. An unresolved strict inequality or
    /// unrepresentable finite enclosure returns an error.
    pub fn explicit(integer_cutoff_c: u64, modes: usize, cutoff_t: &Float) -> Result<Self> {
        if integer_cutoff_c <= 1 || modes == 0 {
            bail!("archimedean tail budget requires integer c > 1 and N >= 1");
        }
        let p = cutoff_t.prec();
        if !(64..=1_000_000).contains(&p) || !cutoff_t.is_finite() || cutoff_t <= &7 {
            bail!("archimedean tail budget requires finite T > 7 and supported precision");
        }
        let working = p + 32;
        let one = I::from_i64(1, working);
        let two = I::from_i64(2, working);
        let n = I::from_u64(u64::try_from(modes)?, working);
        let log_cutoff = I::from_u64(integer_cutoff_c, working).ln()?;
        let pi = I::pi(working);
        let rho = two.mul(&pi).div(&log_cutoff)?;
        let band = rho.mul(&n);
        band.validate()?;
        let theorem_threshold = band.upper().clone().max(&Float::with_val(working, 7));
        if cutoff_t <= &theorem_threshold {
            bail!("strict theorem threshold T > max(rho*N,7) is not established");
        }
        let t = I::from_float(cutoff_t, working)?;
        let denominator = t.sub(&band);
        if !denominator.is_strictly_positive() {
            bail!("positive tail-budget denominator is unresolved");
        }
        let first = t.ln()?.div(&denominator)?;
        // log(T/(T-b)) = log1p(b/(T-b)); this keeps a small second term
        // enclosed tightly without subtracting nearly equal logarithms.
        let ratio = band.div(&denominator)?;
        let mut lower = ratio.lower().clone();
        let mut upper = ratio.upper().clone();
        lower.ln_1p_round(Round::Down);
        upper.ln_1p_round(Round::Up);
        let second = I::new(lower, upper)?.div(&band)?;
        let dimension = two.mul(&n).add(&one);
        let prefactor = two.mul(&dimension).mul(&rho).div(&pi.square())?;
        let enclosure = prefactor.mul(&first.add(&second));
        enclosure.validate()?;
        let upper_bound = Float::with_val_round(p, enclosure.upper(), Round::Up).0;
        if !upper_bound.is_finite() || upper_bound <= 0 {
            bail!("finite positive tail upper bound is unrepresentable");
        }
        Ok(Self {
            integer_cutoff_c,
            log_cutoff,
            rho,
            modes,
            cutoff_t: cutoff_t.clone(),
            theorem_threshold,
            upper_bound,
        })
    }

    /// Exact integer prime cutoff used by this budget.
    pub fn integer_cutoff_c(&self) -> u64 {
        self.integer_cutoff_c
    }
    /// Outward enclosure of log(c), at input precision plus 32 guard bits.
    pub fn log_cutoff(&self) -> &I {
        &self.log_cutoff
    }
    /// Outward enclosure of 2*pi/log(c).
    pub fn rho(&self) -> &I {
        &self.rho
    }
    /// Frequency band {-N,...,N}; no truncating integer conversion is used.
    pub fn modes(&self) -> usize {
        self.modes
    }
    /// Exact stored dyadic cutoff T.
    pub fn cutoff_t(&self) -> &Float {
        &self.cutoff_t
    }
    /// Upward enclosure of max(rho*N,7), below the admitted cutoff T.
    pub fn theorem_threshold(&self) -> &Float {
        &self.theorem_threshold
    }
    /// Outward upper bound for the explicit elementary budget, rounded upward
    /// to the input precision. This also bounds the theorem's positive tail.
    pub fn upper_bound(&self) -> &Float {
        &self.upper_bound
    }
}

/// Conditional conclusion for a finite-T eigenvalue of the same c,N,T problem.
/// The eigenvalue input must be established separately; this type does not
/// authenticate its matrix, eigenvalue index, or numerical error.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FiniteCutoffDecision {
    CutoffFreePositive,
    CutoffFreeNegative,
    /// The supplied enclosure does not establish either sign. The legacy
    /// point wrapper also uses this outcome for nonfinite point inputs.
    InconclusiveTailBand,
}

/// Apply the theorem to an independently established finite-T eigenvalue
/// enclosure for this budget's c,N,T and the same eigenvalue index.
/// Lower >= 0 proves positivity; upper < -B proves negativity. A rounded
/// eigenvalue measurement must first be given a justified error enclosure.
pub fn finite_cutoff_interval_decision(
    finite_t_eigenvalue: &RationalInterval,
    budget: &ArchimedeanTailBudget,
) -> FiniteCutoffDecision {
    if finite_t_eigenvalue.lower() >= &0 {
        FiniteCutoffDecision::CutoffFreePositive
    } else {
        let negative_budget = -budget.upper_bound.clone();
        // MPFR compares directly with the exact rational endpoint. Expanding
        // an extreme dyadic exponent into a huge numerator is unnecessary.
        if negative_budget.partial_cmp(finite_t_eigenvalue.upper())
            == Some(std::cmp::Ordering::Greater)
        {
            FiniteCutoffDecision::CutoffFreeNegative
        } else {
            FiniteCutoffDecision::InconclusiveTailBand
        }
    }
}

/// Checked compatibility calculation treating the point as an exact value.
/// It is a conditional algebraic decision, not validation of an approximate
/// eigenvalue. Use [`finite_cutoff_interval_decision`] with certified endpoints.
pub fn try_finite_cutoff_decision(
    finite_t_eigenvalue: &Float,
    budget: &ArchimedeanTailBudget,
) -> Result<FiniteCutoffDecision> {
    if !finite_t_eigenvalue.is_finite() {
        bail!("finite eigenvalue point required");
    }
    Ok(if finite_t_eigenvalue >= &0 {
        FiniteCutoffDecision::CutoffFreePositive
    } else if finite_t_eigenvalue < &(-budget.upper_bound.clone()) {
        FiniteCutoffDecision::CutoffFreeNegative
    } else {
        FiniteCutoffDecision::InconclusiveTailBand
    })
}

/// Point compatibility wrapper. Nonfinite input is inconclusive. A finite
/// input is interpreted exactly and carries the same external eigenvalue
/// premise as [`try_finite_cutoff_decision`].
pub fn finite_cutoff_decision(
    finite_t_eigenvalue: &Float,
    budget: &ArchimedeanTailBudget,
) -> FiniteCutoffDecision {
    try_finite_cutoff_decision(finite_t_eigenvalue, budget)
        .unwrap_or(FiniteCutoffDecision::InconclusiveTailBand)
}
