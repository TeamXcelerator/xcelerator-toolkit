//! Optional third inverse moment of the retained innovation Gram matrix.
//! The two-mode fit is a model, never an identification of the full spectrum.
use super::{lossless_decimal, PairwiseScratch};
use anyhow::{bail, Result};
use rayon::prelude::*;
use rug::{Assign, Float};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThirdInverseMoment {
    pub inverse_cube_trace: String,
    pub inverse_cube_trace_increment: String,
    pub gram_border_quadratic_form: String,
    pub normalized_gram_square_trace: String,
    pub inverse_square_trace_discrepancy: String,
    pub smallest_eigenvalue_lower_estimate: String,
    pub smallest_eigenvalue_upper_estimate: String,
    pub two_mode_fit: TwoModeMomentFit,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TwoModeMomentStatus {
    SingleDimension,
    Resolved,
    UnresolvedSeparation,
    OutsideTwoModeRange,
    UnresolvedArithmetic,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TwoModeMomentFit {
    pub status: TwoModeMomentStatus,
    pub smallest_eigenvalue_estimate: Option<String>,
    pub second_eigenvalue_estimate: Option<String>,
    pub gap_ratio_estimate: Option<String>,
    /// (T1 - (mu1 + mu2)) / T1. A computed model-closure residual, not proof.
    pub relative_trace_closure_residual: Option<String>,
}
impl TwoModeMomentFit {
    fn unresolved(status: TwoModeMomentStatus) -> Self {
        Self {
            status,
            smallest_eigenvalue_estimate: None,
            second_eigenvalue_estimate: None,
            gap_ratio_estimate: None,
            relative_trace_closure_residual: None,
        }
    }
}
/// Solve mu1^2+mu2^2=T2, mu1^3+mu2^3=T3 under an exactly two-mode model.
/// A general positive spectrum is not determined by three moments. No clipping
/// repairs an unresolved domain, double root or finite-precision separation.
pub(super) fn fit_two_modes(
    t1: &Float,
    t2: &Float,
    t3: &Float,
    dimension: usize,
) -> TwoModeMomentFit {
    let p = t1.prec();
    let cube_root = t3.clone().cbrt();
    if dimension == 1 {
        let mut closure = t1.clone() - &cube_root;
        closure /= t1;
        return TwoModeMomentFit {
            status: TwoModeMomentStatus::SingleDimension,
            smallest_eigenvalue_estimate: Some(lossless_decimal(
                &(Float::with_val(p, 1) / cube_root),
            )),
            second_eigenvalue_estimate: None,
            gap_ratio_estimate: None,
            relative_trace_closure_residual: Some(lossless_decimal(&closure)),
        };
    }
    let root = t2.clone().sqrt();
    let mut q = t3.clone() / t2;
    q /= &root;
    if !q.is_finite() || q <= 0 || q > 1 {
        return TwoModeMomentFit::unresolved(TwoModeMomentStatus::UnresolvedArithmetic);
    }
    if q == 1 {
        return TwoModeMomentFit::unresolved(TwoModeMomentStatus::UnresolvedSeparation);
    }
    if q.clone().square() < Float::with_val(p, 0.5) {
        return TwoModeMomentFit::unresolved(TwoModeMomentStatus::OutsideTwoModeRange);
    }
    // For s=mu1+mu2, x=s/sqrt(T2) is the root in [1,sqrt(2)] of
    // x^3-3x+2q=0. This branch is 2*cos(acos(-q)/3).
    let mut angle = (-q).acos();
    angle /= 3;
    let mut total = angle.cos();
    total *= 2;
    total *= &root;
    let mut discriminant = t2.clone() * 2_i32;
    discriminant -= total.clone().square();
    if !discriminant.is_finite() || discriminant < 0 {
        return TwoModeMomentFit::unresolved(TwoModeMomentStatus::UnresolvedArithmetic);
    }
    let mut mu1 = total.clone() + discriminant.sqrt();
    mu1 /= 2;
    // Product/large-root avoids subtracting nearly equal roots for mu2.
    let mut mu2 = total.clone().square() - t2;
    mu2 /= 2;
    mu2 /= &mu1;
    if !mu1.is_finite() || !mu2.is_finite() || mu2 <= 0 || mu1 < mu2 {
        return TwoModeMomentFit::unresolved(TwoModeMomentStatus::UnresolvedSeparation);
    }
    let mut closure = t1.clone() - &total;
    closure /= t1;
    TwoModeMomentFit {
        status: TwoModeMomentStatus::Resolved,
        smallest_eigenvalue_estimate: Some(lossless_decimal(&(Float::with_val(p, 1) / &mu1))),
        second_eigenvalue_estimate: Some(lossless_decimal(&(Float::with_val(p, 1) / &mu2))),
        gap_ratio_estimate: Some(lossless_decimal(&(mu2 / mu1))),
        relative_trace_closure_residual: Some(lossless_decimal(&closure)),
    }
}

pub(super) struct CubeAccumulator {
    pub pivot_roots: Vec<Float>,
    gram: Vec<Vec<Float>>,
    cube_trace: Float,
    square_trace: Float,
    leaves: Vec<Float>,
    workers: Vec<PairwiseScratch>,
    scratch: PairwiseScratch,
    precision: u32,
}
impl CubeAccumulator {
    pub fn new(p: u32, pivots: &[Float]) -> Self {
        Self {
            pivot_roots: pivots.iter().map(|v| v.clone().sqrt()).collect(),
            gram: Vec::with_capacity(pivots.len()),
            cube_trace: Float::with_val(p, 0),
            square_trace: Float::with_val(p, 0),
            leaves: vec![Float::with_val(p, 0); pivots.len()],
            workers: (0..pivots.len().div_ceil(32))
                .map(|_| PairwiseScratch::new(p))
                .collect(),
            scratch: PairwiseScratch::new(p),
            precision: p,
        }
    }
    pub fn append(
        &mut self,
        row: Vec<Float>,
        t1: &Float,
        t2: &Float,
    ) -> Result<ThirdInverseMoment> {
        let j = self.gram.len();
        let diagonal = &row[j];
        let square = self.scratch.sum_by(j, |i, value| {
            value.assign(&row[i]);
            value.square_mut();
        });
        let term = |i: usize, value: &mut Float, local: &mut PairwiseScratch| {
            // g^T G g = sum_i (G_ii*g_i^2 + 2*g_i*sum_{l<i}G_il*g_l).
            // Stored triangular rows and a fixed indexed tree avoid duplicates.
            value.assign(local.sum_by(i, |l, product| {
                product.assign(&self.gram[i][l]);
                *product *= &row[l];
            }));
            *value *= &row[i];
            *value *= 2;
            let mut diagonal_term = row[i].clone().square();
            diagonal_term *= &self.gram[i][i];
            *value += diagonal_term;
        };
        if j >= 128 && rayon::current_num_threads() > 1 {
            self.leaves[..j]
                .par_chunks_mut(32)
                .zip(self.workers.par_iter_mut())
                .enumerate()
                .for_each(|(block, (leaves, local))| {
                    for (offset, value) in leaves.iter_mut().enumerate() {
                        term(block * 32 + offset, value, local);
                    }
                });
        } else {
            for (i, value) in self.leaves[..j].iter_mut().enumerate() {
                term(i, value, &mut self.workers[0]);
            }
        }
        let quadratic = self
            .scratch
            .sum_by(j, |i, value| value.assign(&self.leaves[i]));
        let mut delta = quadratic.clone() * 3_i32;
        let mut mixed = square.clone() * diagonal;
        mixed *= 3;
        delta += mixed;
        let mut diagonal_cube = diagonal.clone().square();
        diagonal_cube *= diagonal;
        delta += diagonal_cube;
        self.cube_trace += &delta;
        self.square_trace += square * 2_i32 + diagonal.clone().square();
        if !self.cube_trace.is_finite() || self.cube_trace <= 0 || !quadratic.is_finite() {
            bail!("unresolved third inverse moment at prefix {}", j + 1);
        }
        let lower = Float::with_val(self.precision, 1) / self.cube_trace.clone().cbrt();
        let upper = t2.clone() / &self.cube_trace;
        let discrepancy = self.square_trace.clone() - t2;
        let report = ThirdInverseMoment {
            inverse_cube_trace: lossless_decimal(&self.cube_trace),
            inverse_cube_trace_increment: lossless_decimal(&delta),
            gram_border_quadratic_form: lossless_decimal(&quadratic),
            normalized_gram_square_trace: lossless_decimal(&self.square_trace),
            inverse_square_trace_discrepancy: lossless_decimal(&discrepancy),
            smallest_eigenvalue_lower_estimate: lossless_decimal(&lower),
            smallest_eigenvalue_upper_estimate: lossless_decimal(&upper),
            two_mode_fit: fit_two_modes(t1, t2, &self.cube_trace, j + 1),
        };
        self.gram.push(row);
        Ok(report)
    }
}

impl ThirdInverseMoment {
    /// Validate serialized scalar and model consistency at the recorded precision.
    /// This does not replay the matrix factorization or certify the moment itself.
    pub fn validate_for_moments(&self, t1: &Float, t2: &Float, dimension: usize) -> Result<()> {
        let p = t1.prec();
        if t2.prec() != p || !t1.is_finite() || !t2.is_finite() || t1 <= &0 || t2 <= &0 {
            bail!("invalid source inverse moments for validation");
        }
        let parse = |s: &str| -> Result<Float> {
            let value = Float::with_val(p, Float::parse(s)?);
            if !value.is_finite() {
                bail!("nonfinite third-moment scalar");
            }
            Ok(value)
        };
        let t3 = parse(&self.inverse_cube_trace)?;
        let square = parse(&self.normalized_gram_square_trace)?;
        parse(&self.inverse_cube_trace_increment)?;
        parse(&self.gram_border_quadratic_form)?;
        if t3 <= 0
            || square <= 0
            || dimension == 0
            || self.inverse_square_trace_discrepancy != lossless_decimal(&(square - t2))
            || self.smallest_eigenvalue_lower_estimate
                != lossless_decimal(&(Float::with_val(p, 1) / t3.clone().cbrt()))
            || self.smallest_eigenvalue_upper_estimate != lossless_decimal(&(t2.clone() / &t3))
            || self.two_mode_fit != fit_two_modes(t1, t2, &t3, dimension)
        {
            bail!("inconsistent third-moment or two-mode diagnostics");
        }
        Ok(())
    }
}
