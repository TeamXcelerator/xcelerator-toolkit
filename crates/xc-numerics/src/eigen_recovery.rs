//! Source-bound eigenvector recovery: a directed residual and exclusion of every other index.
use super::{HpTridiagonalEigenvalueEnclosure, TridiagEigvecOptions, TridiagSolver};
use crate::{linalg, mpfr_interval::MpfrInterval as I, symmetric_inertia};
use anyhow::Result;
use rug::{float::Round, ops::Pow, Float};

pub const DENSE_EIGENVECTOR_SEMANTICS: &str =
    "dense-eigenvector-requested-source-rounding-exact-count-scaling-directed-index-gap-angle-v3";
/// Recoverable arithmetic failures are distinct from invalid requests.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HpEigenvectorRecoveryFailure {
    InvalidConfiguration(String),
    UnresolvedEigenspace(String),
    IterationLimit { index: usize, maximum_steps: usize },
}
impl std::fmt::Display for HpEigenvectorRecoveryFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfiguration(s) => write!(f, "invalid eigenvector recovery: {s}"),
            Self::UnresolvedEigenspace(s) => write!(f, "unresolved eigenvector eigenspace: {s}"),
            Self::IterationLimit { index, maximum_steps } => write!(f, "selected index {index} did not meet its directed angle bound in {maximum_steps} inverse iterations"),
        }
    }
}
impl std::error::Error for HpEigenvectorRecoveryFailure {}
fn unresolved(s: impl Into<String>) -> anyhow::Error {
    HpEigenvectorRecoveryFailure::UnresolvedEigenspace(s.into()).into()
}
/// Evidence concerns the working symmetric matrix and the returned vector.
/// Direct recovery rounds source entries once to the requested precision;
/// selected tridiagonal recovery retains the exact source used by its counts.
/// `sine_angle_upper_bound` bounds sin(angle) to the unique selected eigenspace.
/// It is not an assertion of `precision_bits` accurate vector bits. In a close
/// pair the returned-vector rounding floor may dominate; that fact is explicit.
#[derive(Clone, Debug, PartialEq)]
pub struct HpEigenvectorRecovery {
    pub eigenvalue: Float,
    pub eigenvector: Vec<Float>,
    pub index: usize,
    /// Count-proven isolating window; not an eigenvalue error estimate.
    pub enclosure: HpTridiagonalEigenvalueEnclosure,
    pub residual_upper_bound: Float,
    pub separation_lower_bound: Float,
    pub sine_angle_upper_bound: Float,
    pub angle_acceptance_bound: Float,
    pub rounding_floor_limited: bool,
    pub working_precision_bits: u32,
    pub iterations: usize,
    pub seed_solves: usize,
}
#[derive(Clone, Copy)]
enum Matrix<'a> {
    Tridiagonal(&'a [Float], &'a [Float]),
    Dense(&'a [Float], usize),
}
impl<'a> Matrix<'a> {
    fn n(self) -> usize {
        match self {
            Self::Tridiagonal(d, _) => d.len(),
            Self::Dense(_, n) => n,
        }
    }
    fn entries(self) -> Box<dyn Iterator<Item = &'a Float> + 'a> {
        match self {
            Self::Tridiagonal(d, e) => Box::new(d.iter().chain(e)),
            Self::Dense(a, _) => Box::new(a.iter()),
        }
    }
    fn validate(self, p: u32) -> Result<()> {
        let n = self.n();
        let shape = match self {
            Self::Tridiagonal(d, e) => e.len().checked_add(1) == Some(d.len()),
            Self::Dense(a, n) => n.checked_mul(n) == Some(a.len()),
        };
        if n == 0
            || !shape
            || !(33..=1_000_000).contains(&p)
            || self.entries().any(|x| !x.is_finite())
        {
            return Err(HpEigenvectorRecoveryFailure::InvalidConfiguration(
                "finite square symmetric storage and a requested precision (33..=1000000) required"
                    .into(),
            )
            .into());
        }
        if let Self::Dense(a, n) = self {
            if (0..n).any(|i| (0..i).any(|j| a[i * n + j] != a[j * n + i])) {
                return Err(HpEigenvectorRecoveryFailure::InvalidConfiguration(
                    "asymmetric matrix".into(),
                )
                .into());
            }
        }
        Ok(())
    }
    fn scale(self, p: u32) -> Result<Float> {
        let n = self.n();
        let mut result = Float::with_val(p, 0);
        for i in 0..n {
            let mut row = Float::with_val(p, 0);
            self.row(i, |_, a| {
                row = Float::with_val_round(p, &row + a.clone().abs(), Round::Up).0;
            });
            if row > result {
                result = row;
            }
        }
        if !result.is_finite() {
            return Err(unresolved("matrix norm bound exceeds the finite range"));
        }
        if result.is_zero() {
            result = Float::with_val(p, 1);
        }
        Ok(result)
    }
    fn row(self, i: usize, mut visit: impl FnMut(usize, &Float)) {
        match self {
            Self::Dense(a, n) => {
                for j in 0..n {
                    visit(j, &a[i * n + j]);
                }
            }
            Self::Tridiagonal(d, e) => {
                visit(i, &d[i]);
                if i > 0 {
                    visit(i - 1, &e[i - 1]);
                }
                if i + 1 < d.len() {
                    visit(i + 1, &e[i]);
                }
            }
        }
    }
    fn guide(self, p: u32) -> Result<Vec<Float>> {
        match self {
            Self::Dense(a, n) => super::dense_symmetric_eigenvalues_hp_stable(a, n, p),
            Self::Tridiagonal(d, e) => super::tridiag_eigenvalues_hp(d, e, p),
        }
    }
    fn count(self, x: &Float, p: u32) -> Result<usize> {
        // A common positive binary scaling preserves the exact signs of A-xI.
        // Do not construct a rounded shifted source: scale entries and x separately,
        // retaining every source bit, before the directed count subtracts them.
        let exponent = self
            .entries()
            .chain(std::iter::once(x))
            .filter_map(Float::get_exp)
            .max()
            .map_or(0, i64::from);
        let scaled: Vec<_> = self
            .entries()
            .map(|v| count_scale(v, exponent, p))
            .collect::<Result<_>>()?;
        let threshold = count_scale(x, exponent, p)?;
        match self {
            Self::Tridiagonal(d, _) => super::tridiag_sturm_count_below_hp(
                &scaled[..d.len()],
                &scaled[d.len()..],
                &threshold,
                p,
            ),
            Self::Dense(_, n) => match symmetric_inertia::point_matrix_inertia_at(
                &scaled,
                n,
                &threshold,
                p,
                8 * 1024 * 1024 * 1024,
            )? {
                symmetric_inertia::MpfrInertiaResult::Conclusive { negative, .. } => Ok(negative),
                symmetric_inertia::MpfrInertiaResult::Inconclusive { reason, .. } => {
                    Err(unresolved(reason))
                }
            },
        }
    }
    fn factor(self, shift: &Float, p: u32, solver: TridiagSolver) -> Result<Factor> {
        match self {
            Self::Tridiagonal(d, e) if !matches!(solver, TridiagSolver::Dense) => {
                let diagonal: Vec<_> = d.iter().map(|x| Float::with_val(p, x - shift)).collect();
                let off: Vec<_> = e.iter().map(|x| Float::with_val(p, x)).collect();
                Ok(Factor::Tridiagonal(linalg::tridiag_lu_factor_hp(
                    &off, &diagonal, &off, p,
                )?))
            }
            _ => {
                let n = self.n();
                let mut a = vec![Float::with_val(p, 0); n * n];
                for i in 0..n {
                    self.row(i, |j, x| a[i * n + j] = Float::with_val(p, x));
                    a[i * n + i] -= shift;
                }
                Ok(Factor::Dense(linalg::lu_factor(&a, n)?))
            }
        }
    }
}
fn count_scale(x: &Float, exponent: i64, p: u32) -> Result<Float> {
    if x.prec() > p {
        return Err(unresolved("count scaling would discard source bits"));
    }
    let amount = u32::try_from(exponent.unsigned_abs())?;
    let mut scaled = Float::with_val(p, x);
    if exponent >= 0 {
        scaled >>= amount;
    } else {
        scaled <<= amount;
    }
    if !scaled.is_finite() || (!x.is_zero() && scaled.is_zero()) {
        return Err(unresolved(
            "exact count preconditioning exceeds the finite exponent range",
        ));
    }
    let mut reverse = scaled.clone();
    if exponent >= 0 {
        reverse <<= amount;
    } else {
        reverse >>= amount;
    }
    if reverse != *x {
        return Err(unresolved(
            "exact count preconditioning loses a source component",
        ));
    }
    Ok(scaled)
}

enum Factor {
    Dense(linalg::LuFactors),
    Tridiagonal(linalg::TridiagLuFactors),
}
impl Factor {
    fn solve(&self, v: &[Float], p: u32) -> Result<Vec<Float>> {
        match self {
            Self::Dense(f) => linalg::try_lu_solve(f, v, v.len(), p),
            Self::Tridiagonal(f) => linalg::tridiag_lu_solve_pivoted_hp(f, v, p),
        }
    }
}
fn midpoint(a: &Float, b: &Float, p: u32) -> Float {
    let mut x = Float::with_val(p, a);
    x /= 2;
    let mut y = Float::with_val(p, b);
    y /= 2;
    x += y;
    x
}

struct RecoveryGuide {
    scale: Float,
    values: Vec<Float>,
    precision_bits: u32,
}

#[cfg(test)]
thread_local! {
    static GUIDE_COMPUTATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn validate_request(
    n: usize,
    requested: Option<usize>,
    value: Option<&Float>,
    uncertainty: Option<&Float>,
    opts: TridiagEigvecOptions,
) -> Result<()> {
    if opts.max_steps == 0
        || requested.is_some_and(|k| k >= n)
        || value.is_some_and(|x| !x.is_finite())
        || uncertainty.is_some_and(|x| !x.is_finite() || x < &0)
    {
        return Err(HpEigenvectorRecoveryFailure::InvalidConfiguration(
            "invalid index, eigenvalue, uncertainty, or zero iteration budget".into(),
        )
        .into());
    }
    Ok(())
}

fn prepare_guide(matrix: Matrix<'_>, p: u32, opts: TridiagEigvecOptions) -> Result<RecoveryGuide> {
    let n = matrix.n();
    let source_bits = matrix.entries().map(Float::prec).max().unwrap_or(p);
    let w = (p * 2 + 64)
        .max(source_bits.saturating_add(32))
        .min(1_000_000);
    // Includes QR, retained factors, interval inertia, and vector scratch.
    let entries =
        if matches!(matrix, Matrix::Dense(..)) || matches!(opts.solver, TridiagSolver::Dense) {
            8u128 * n as u128 * n as u128 + 32 * n as u128
        } else {
            64 * n as u128
        };
    if entries * (64 + u128::from(w).div_ceil(8)) > 8 * 1024 * 1024 * 1024 {
        return Err(unresolved(
            "eigenvector working storage exceeds the 8 GiB admission bound",
        ));
    }
    let scale = matrix.scale(w)?;
    #[cfg(test)]
    GUIDE_COMPUTATIONS.with(|count| count.set(count.get() + 1));
    let values = matrix.guide(w)?;
    Ok(RecoveryGuide {
        scale,
        values,
        precision_bits: w,
    })
}

/// Reuse one working-precision guide when recovering several vectors of the
/// same tridiagonal source. Construction owns a copy rounded to `precision_bits`;
/// the source, precision, and solver options cannot change during its lifetime.
///
/// The first valid request prepares the guide. Each request still proves its
/// index window and checks its returned vector's directed residual and angle.
/// The guide is not a certificate and is never supplied by the caller. Failed
/// guide preparation is not remembered. No persistent cache identity is added.
pub struct TridiagonalEigenvectorRecoveryHp {
    source: Vec<Float>,
    dimension: usize,
    precision_bits: u32,
    options: TridiagEigvecOptions,
    guide: Option<RecoveryGuide>,
}

impl TridiagonalEigenvectorRecoveryHp {
    pub fn new(
        d: &[Float],
        e: &[Float],
        precision_bits: u32,
        options: TridiagEigvecOptions,
    ) -> Result<Self> {
        let matrix = Matrix::Tridiagonal(d, e);
        matrix.validate(precision_bits)?;
        let source: Vec<_> = matrix
            .entries()
            .map(|x| Float::with_val(precision_bits, x))
            .collect();
        if source.iter().any(|x| !x.is_finite()) {
            return Err(unresolved("source rounding exceeds the finite range"));
        }
        Ok(Self {
            source,
            dimension: d.len(),
            precision_bits,
            options,
            guide: None,
        })
    }

    /// Recover the member identified by `value`, with the same arithmetic and
    /// request checks as [`tridiag_eigenvector_for_value_detailed_hp`].
    pub fn recover_for_value(
        &mut self,
        value: &Float,
        uncertainty: Option<&Float>,
    ) -> Result<HpEigenvectorRecovery> {
        validate_request(self.dimension, None, Some(value), uncertainty, self.options)?;
        let matrix = Matrix::Tridiagonal(
            &self.source[..self.dimension],
            &self.source[self.dimension..],
        );
        if self.guide.is_none() {
            let started = std::time::Instant::now();
            self.guide = Some(prepare_guide(matrix, self.precision_bits, self.options)?);
            xc_core::progress_message!("[HP eigenvector] shared tridiagonal guide: dimension={}, precision={} bits, computed in {:.3}s", self.dimension, self.guide.as_ref().expect("prepared guide").precision_bits, started.elapsed().as_secs_f64());
        }
        recover_with_guide(
            matrix,
            None,
            Some(value),
            uncertainty,
            self.precision_bits,
            self.options,
            self.guide.as_ref().expect("prepared guide"),
        )
    }
}

fn recover(
    matrix: Matrix<'_>,
    requested: Option<usize>,
    value: Option<&Float>,
    uncertainty: Option<&Float>,
    p: u32,
    opts: TridiagEigvecOptions,
    round_source: bool,
) -> Result<HpEigenvectorRecovery> {
    matrix.validate(p)?;
    // Guide, counts, factors, and residual evidence must use the same source.
    let rounded: Vec<_> = if round_source {
        matrix.entries().map(|x| Float::with_val(p, x)).collect()
    } else {
        Vec::new()
    };
    if rounded.iter().any(|x| !x.is_finite()) {
        return Err(unresolved("source rounding exceeds the finite range"));
    }
    let matrix = if round_source {
        match matrix {
            Matrix::Dense(_, n) => Matrix::Dense(&rounded, n),
            Matrix::Tridiagonal(d, _) => {
                Matrix::Tridiagonal(&rounded[..d.len()], &rounded[d.len()..])
            }
        }
    } else {
        matrix
    };
    let n = matrix.n();
    validate_request(n, requested, value, uncertainty, opts)?;
    let guide = prepare_guide(matrix, p, opts)?;
    recover_with_guide(matrix, requested, value, uncertainty, p, opts, &guide)
}

fn recover_with_guide(
    matrix: Matrix<'_>,
    requested: Option<usize>,
    value: Option<&Float>,
    uncertainty: Option<&Float>,
    p: u32,
    opts: TridiagEigvecOptions,
    guide: &RecoveryGuide,
) -> Result<HpEigenvectorRecovery> {
    let n = matrix.n();
    let w = guide.precision_bits;
    let scale = guide.scale.clone();
    let values = &guide.values;
    let index = if let Some(k) = requested {
        k
    } else {
        let value = value.ok_or_else(|| unresolved("missing requested member"))?;
        let mut radius = Float::with_val(w, 2).pow(-(p as i32));
        radius *= 32usize.saturating_mul(n);
        radius *= &scale;
        if let Some(u) = uncertainty {
            radius += u;
        }
        let candidates: Vec<_> = values
            .iter()
            .enumerate()
            .filter(|(_, x)| Float::with_val(w, *x - value).abs() <= radius)
            .map(|(i, _)| i)
            .collect();
        if candidates.len() != 1 {
            return Err(unresolved(format!(
                "requested value does not identify exactly one eigenvalue ({} candidate members)",
                candidates.len()
            )));
        }
        candidates[0]
    };
    let mut lower = if index > 0 {
        midpoint(&values[index - 1], &values[index], w)
    } else {
        let mut x = scale.clone();
        x *= -2;
        x
    };
    let mut upper = if index + 1 < n {
        midpoint(&values[index], &values[index + 1], w)
    } else {
        let mut x = scale.clone();
        x *= 2;
        x
    };
    if lower >= values[index] || upper <= values[index] {
        return Err(unresolved(format!(
            "selected index {index} has no separated working-precision neighbors"
        )));
    }
    // A computed spectrum only proposes boundaries. Directed counts establish
    // that precisely the requested member is inside and all others outside.
    let mut proven = false;
    for _ in 0..8 {
        if matrix.count(&lower, w).ok() == Some(index)
            && matrix.count(&upper, w).ok() == Some(index + 1)
        {
            proven = true;
            break;
        }
        lower.next_down();
        upper.next_up();
    }
    if !proven {
        return Err(unresolved(format!(
            "directed counts could not isolate selected index {index}"
        )));
    }
    let mut shift = values[index].clone();
    let factor = match matrix.factor(&shift, w, opts.solver) {
        Ok(f) => f,
        Err(_) => {
            let mut perturb = Float::with_val(w, 2).pow(-(w as i32) + 8);
            perturb *= &scale;
            shift += perturb;
            if shift <= lower || shift >= upper {
                return Err(unresolved(
                    "representable inverse shift leaves selected window",
                ));
            }
            matrix
                .factor(&shift, w, opts.solver)
                .map_err(|e| unresolved(format!("inverse factorization unresolved: {e}")))?
        }
    };
    let mut seed_solves = 0;
    let mut v = match matrix {
        Matrix::Tridiagonal(_, e) => (0..n)
            .map(|i| Float::with_val(w, u32::from(i == 0 || e[i - 1].is_zero())))
            .collect::<Vec<_>>(),
        Matrix::Dense(..) => {
            // Every coordinate is tested: an even initial vector cannot hide
            // an odd member. Select the largest inverse column without storing
            // the inverse; setup is n solves and O(n) additional storage.
            let mut best = Vec::new();
            let mut best_scale = Float::with_val(w, 0);
            for j in 0..n {
                let mut basis = vec![Float::with_val(w, 0); n];
                basis[j] = Float::with_val(w, 1);
                let column = factor.solve(&basis, w)?;
                seed_solves += 1;
                let magnitude = column
                    .iter()
                    .map(|x| x.clone().abs())
                    .max_by(Float::total_cmp)
                    .ok_or_else(|| unresolved("empty inverse column"))?;
                if magnitude > best_scale {
                    best_scale = magnitude;
                    best = column;
                }
            }
            best
        }
    };
    linalg::try_normalize_l2(&mut v)?;
    let mut last = None;
    for iteration in 1..=opts.max_steps {
        v = factor.solve(&v, w)?;
        linalg::try_normalize_l2(&mut v)?;
        // Admission always concerns the rounded vector actually returned.
        let rounded: Vec<_> = v.iter().map(|x| Float::with_val(p, x)).collect();
        if let Some(mut report) = assess(matrix, &rounded, &lower, &upper, &scale, index, p, w)? {
            report.iterations = iteration;
            report.seed_solves = seed_solves;
            last = Some(report);
            if opts.early_termination {
                return Ok(last.expect("just assigned"));
            }
        } else {
            last = None;
        }
    }
    last.ok_or_else(|| {
        HpEigenvectorRecoveryFailure::IterationLimit {
            index,
            maximum_steps: opts.max_steps,
        }
        .into()
    })
}
#[allow(clippy::too_many_arguments)]
fn assess(
    matrix: Matrix<'_>,
    v: &[Float],
    lower: &Float,
    upper: &Float,
    scale: &Float,
    index: usize,
    p: u32,
    w: u32,
) -> Result<Option<HpEigenvectorRecovery>> {
    let n = v.len();
    let s = I::from_float(scale, w)?;
    let points = v
        .iter()
        .map(|x| I::from_float(x, w))
        .collect::<Result<Vec<_>, _>>()?;
    let mut norm = I::from_i64(0, w);
    let mut numerator = I::from_i64(0, w);
    let mut actions = Vec::with_capacity(n);
    for i in 0..n {
        let mut av = I::from_i64(0, w);
        let mut error = None;
        matrix.row(i, |j, a| {
            match I::from_float(a, w).and_then(|a| a.div(&s)) {
                Ok(a) => av = av.add(&a.mul(&points[j])),
                Err(e) => error = Some(e),
            }
        });
        if let Some(e) = error {
            return Err(e.into());
        }
        av.validate()?;
        norm = norm.add(&points[i].square());
        numerator = numerator.add(&points[i].mul(&av));
        actions.push(av);
    }
    let quotient = numerator.div(&norm)?.mul(&s);
    quotient.validate()?;
    let rho = Float::with_val(p, quotient.midpoint_point().lower());
    let ri = I::from_float(&rho, w)?;
    if &rho <= lower || &rho >= upper {
        return Ok(None);
    }
    let separation = ri
        .sub(&I::from_float(lower, w)?)
        .lower()
        .clone()
        .min(I::from_float(upper, w)?.sub(&ri).lower());
    if separation <= 0 {
        return Ok(None);
    }
    let relative_rho = ri.div(&s)?;
    let mut residual = I::from_i64(0, w);
    for i in 0..n {
        residual = residual.add(&actions[i].sub(&relative_rho.mul(&points[i])).square());
    }
    let residual = residual.sqrt()?.div(&norm.sqrt()?)?;
    let delta = I::from_float(&separation, w)?.div(&s)?;
    let angle = residual.div(&delta)?;
    angle.validate()?;
    let mut floor = Float::with_val(w, 2).pow(-(p as i32));
    floor *= 8usize.saturating_mul(n);
    let floor = I::from_float(&floor, w)?.div(&delta)?.upper().clone();
    let desired = Float::with_val(w, 2).pow(-((p / 2) as i32));
    let rounding_floor_limited = floor > desired;
    let target = floor.max(&desired).min(&Float::with_val(w, 2).pow(-8i32));
    if angle.upper() > &target {
        return Ok(None);
    }
    let residual_abs = residual.mul(&s);
    residual_abs.validate()?;
    Ok(Some(HpEigenvectorRecovery {
        eigenvalue: rho,
        eigenvector: v.to_vec(),
        index,
        enclosure: HpTridiagonalEigenvalueEnclosure {
            index,
            lower: lower.clone(),
            upper: upper.clone(),
            lower_count: index,
            upper_count: index + 1,
            iterations: 0,
        },
        residual_upper_bound: residual_abs.upper().clone(),
        separation_lower_bound: separation,
        sine_angle_upper_bound: angle.upper().clone(),
        angle_acceptance_bound: target,
        rounding_floor_limited,
        working_precision_bits: w,
        iterations: 0,
        seed_solves: 0,
    }))
}
/// Recheck a stored tridiagonal vector with the recovery path's directed
/// residual/separation gate, without a spectrum solve or inverse iteration.
/// The supplied isolating window is proved again by exact-source Sturm counts.
/// This checks the rounded source matrix and vector, not continuum accuracy.
#[allow(clippy::too_many_arguments)]
pub fn validate_tridiag_eigenvector_hp(
    d: &[Float],
    e: &[Float],
    vector: &[Float],
    index: usize,
    lower: &Float,
    upper: &Float,
    p: u32,
) -> Result<HpEigenvectorRecovery> {
    // Reject invalid MPFR precision before invoking Float::with_val, which
    // otherwise panics instead of returning the public API's error result.
    Matrix::Tridiagonal(d, e).validate(p)?;
    let rounded_d: Vec<_> = d.iter().map(|x| Float::with_val(p, x)).collect();
    let rounded_e: Vec<_> = e.iter().map(|x| Float::with_val(p, x)).collect();
    let matrix = Matrix::Tridiagonal(&rounded_d, &rounded_e);
    matrix.validate(p)?;
    anyhow::ensure!(
        vector.len() == d.len()
            && index < d.len()
            && vector.iter().all(Float::is_finite)
            && lower.is_finite()
            && upper.is_finite()
            && lower < upper,
        "invalid stored eigenvector or isolating window"
    );
    let w = (p * 2 + 64).max(p.saturating_add(32)).min(1_000_000);
    anyhow::ensure!(
        matrix.count(lower, w)? == index && matrix.count(upper, w)? == index + 1,
        "stored eigenvector window does not isolate its source index"
    );
    let scale = matrix.scale(w)?;
    assess(matrix, vector, lower, upper, &scale, index, p, w)?
        .ok_or_else(|| unresolved("stored eigenvector fails directed residual/separation gate"))
}

/// Selected-index recovery of the matrix rounded once to `p` bits, with directed angle evidence.
pub fn dense_symmetric_eigenpair_at_index_hp(
    a: &[Float],
    n: usize,
    index: usize,
    p: u32,
    max_steps: usize,
) -> Result<HpEigenvectorRecovery> {
    recover(
        Matrix::Dense(a, n),
        Some(index),
        None,
        None,
        p,
        TridiagEigvecOptions {
            max_steps,
            early_termination: true,
            solver: TridiagSolver::Dense,
        },
        true,
    )
}
pub fn dense_symmetric_eigenvector_for_value_detailed_hp(
    a: &[Float],
    n: usize,
    value: &Float,
    p: u32,
    max_steps: usize,
) -> Result<HpEigenvectorRecovery> {
    recover(
        Matrix::Dense(a, n),
        None,
        Some(value),
        None,
        p,
        TridiagEigvecOptions {
            max_steps,
            early_termination: true,
            solver: TridiagSolver::Dense,
        },
        true,
    )
}
pub fn tridiag_eigenvector_for_value_detailed_hp(
    d: &[Float],
    e: &[Float],
    value: &Float,
    uncertainty: Option<&Float>,
    p: u32,
    opts: TridiagEigvecOptions,
) -> Result<HpEigenvectorRecovery> {
    recover(
        Matrix::Tridiagonal(d, e),
        None,
        Some(value),
        uncertainty,
        p,
        opts,
        true,
    )
}
pub(super) fn tridiagonal_at_index(
    d: &[Float],
    e: &[Float],
    index: usize,
    p: u32,
    opts: TridiagEigvecOptions,
) -> Result<HpEigenvectorRecovery> {
    recover(
        Matrix::Tridiagonal(d, e),
        Some(index),
        None,
        None,
        p,
        opts,
        false,
    )
}

#[cfg(test)]
mod cached_vector_validation_tests {
    use super::*;

    #[test]
    fn stored_tridiagonal_vector_rejects_invalid_precision_before_rounding() {
        let diagonal = [Float::with_val(64, 1)];
        let vector = [Float::with_val(64, 1)];
        let lower = Float::with_val(64, 0);
        let upper = Float::with_val(64, 2);
        for p in [0, 1, 32, 1_000_001] {
            assert!(
                validate_tridiag_eigenvector_hp(&diagonal, &[], &vector, 0, &lower, &upper, p,)
                    .is_err()
            );
        }
        let report =
            validate_tridiag_eigenvector_hp(&diagonal, &[], &vector, 0, &lower, &upper, 64)
                .unwrap();
        assert_eq!(report.eigenvalue, 1);
        assert_eq!(report.iterations, 0);
        assert_eq!(report.seed_solves, 0);
    }
}

#[cfg(test)]
mod shared_guide_tests {
    use super::*;

    fn assert_identical(a: &HpEigenvectorRecovery, b: &HpEigenvectorRecovery) {
        assert_eq!(a, b);
        let scalars = |r: &HpEigenvectorRecovery| {
            r.eigenvector
                .iter()
                .chain([
                    &r.eigenvalue,
                    &r.enclosure.lower,
                    &r.enclosure.upper,
                    &r.residual_upper_bound,
                    &r.separation_lower_bound,
                    &r.sine_angle_upper_bound,
                    &r.angle_acceptance_bound,
                ])
                .map(|x| (x.prec(), x.is_sign_negative(), x.to_string_radix(16, None)))
                .collect::<Vec<_>>()
        };
        assert_eq!(scalars(a), scalars(b));
    }

    #[test]
    fn shared_guide_matches_fresh_reports_in_any_request_order() {
        for solver in [TridiagSolver::Dense, TridiagSolver::BandedInterleaved] {
            let p = 64;
            let mut d: Vec<_> = (1..=5).map(|x| Float::with_val(160, x)).collect();
            d[1] += Float::with_val(160, 2).pow(-100);
            let e = vec![Float::with_val(160, -0.125); 4];
            let rounded: Vec<_> = d.iter().map(|x| Float::with_val(p, x)).collect();
            let values = super::super::tridiag_eigenvalues_hp(&rounded, &e, p).unwrap();
            let options = TridiagEigvecOptions {
                solver,
                ..Default::default()
            };
            let order = [4, 0, 1, 4];
            let expected: Vec<_> = order
                .iter()
                .map(|&k| {
                    tridiag_eigenvector_for_value_detailed_hp(&d, &e, &values[k], None, p, options)
                        .unwrap()
                })
                .collect();
            let mut context = TridiagonalEigenvectorRecoveryHp::new(&d, &e, p, options).unwrap();
            // Owning the rounded source prevents later caller mutation from
            // pairing a retained guide with a different matrix.
            d[0] += 100;
            GUIDE_COMPUTATIONS.with(|count| count.set(0));
            for (&k, expected) in order.iter().zip(&expected) {
                let actual = context.recover_for_value(&values[k], None).unwrap();
                assert_identical(&actual, expected);
                assert_eq!(actual.working_precision_bits, 2 * p + 64);
            }
            assert_eq!(GUIDE_COMPUTATIONS.with(std::cell::Cell::get), 1);
        }
    }

    #[test]
    fn shared_guide_invalid_requests_do_not_prepare_and_valid_retry_succeeds() {
        let d = [Float::with_val(64, 1), Float::with_val(64, 3)];
        let e = [Float::with_val(64, 0)];
        let options = TridiagEigvecOptions::default();
        let mut context = TridiagonalEigenvectorRecoveryHp::new(&d, &e, 64, options).unwrap();
        GUIDE_COMPUTATIONS.with(|count| count.set(0));
        for (value, uncertainty) in [
            (Float::with_val(64, rug::float::Special::Nan), None),
            (d[0].clone(), Some(Float::with_val(64, -1))),
            (
                d[0].clone(),
                Some(Float::with_val(64, rug::float::Special::Infinity)),
            ),
        ] {
            let shared = context
                .recover_for_value(&value, uncertainty.as_ref())
                .unwrap_err();
            let fresh = tridiag_eigenvector_for_value_detailed_hp(
                &d,
                &e,
                &value,
                uncertainty.as_ref(),
                64,
                options,
            )
            .unwrap_err();
            assert_eq!(
                shared.downcast_ref::<HpEigenvectorRecoveryFailure>(),
                fresh.downcast_ref::<HpEigenvectorRecoveryFailure>()
            );
            assert!(context.guide.is_none());
        }
        assert_eq!(GUIDE_COMPUTATIONS.with(std::cell::Cell::get), 0);
        for (value, uncertainty) in [
            (Float::with_val(64, 9), None),
            (d[0].clone(), Some(Float::with_val(64, 8))),
        ] {
            let shared = context
                .recover_for_value(&value, uncertainty.as_ref())
                .unwrap_err();
            let fresh = tridiag_eigenvector_for_value_detailed_hp(
                &d,
                &e,
                &value,
                uncertainty.as_ref(),
                64,
                options,
            )
            .unwrap_err();
            assert_eq!(shared.to_string(), fresh.to_string());
        }
        let a = context.recover_for_value(&d[0], None).unwrap();
        let b =
            tridiag_eigenvector_for_value_detailed_hp(&d, &e, &d[0], None, 64, options).unwrap();
        assert_identical(&a, &b);
    }

    #[test]
    fn shared_guide_preserves_source_and_option_refusals() {
        let d = [Float::with_val(64, 1)];
        for p in [0, 32, 1_000_001] {
            assert!(TridiagonalEigenvectorRecoveryHp::new(&d, &[], p, Default::default()).is_err());
        }
        assert!(TridiagonalEigenvectorRecoveryHp::new(&d, &d, 64, Default::default()).is_err());
        assert!(TridiagonalEigenvectorRecoveryHp::new(
            &[Float::with_val(64, rug::float::Special::Nan)],
            &[],
            64,
            Default::default()
        )
        .is_err());
        let options = TridiagEigvecOptions {
            max_steps: 0,
            ..Default::default()
        };
        let mut context = TridiagonalEigenvectorRecoveryHp::new(&d, &[], 64, options).unwrap();
        let shared = context.recover_for_value(&d[0], None).unwrap_err();
        let fresh = tridiag_eigenvector_for_value_detailed_hp(&d, &[], &d[0], None, 64, options)
            .unwrap_err();
        assert_eq!(shared.to_string(), fresh.to_string());
        assert!(context.guide.is_none());
    }

    #[test]
    fn shared_guide_repeated_other_members_do_not_reject_simple_member() {
        let d: Vec<_> = [1, 1, 3]
            .into_iter()
            .map(|x| Float::with_val(64, x))
            .collect();
        let e = vec![Float::with_val(64, 0); 2];
        let options = TridiagEigvecOptions::default();
        let mut context = TridiagonalEigenvectorRecoveryHp::new(&d, &e, 64, options).unwrap();
        let shared = context.recover_for_value(&d[0], None).unwrap_err();
        let fresh = tridiag_eigenvector_for_value_detailed_hp(&d, &e, &d[0], None, 64, options)
            .unwrap_err();
        assert_eq!(shared.to_string(), fresh.to_string());
        assert_identical(
            &context.recover_for_value(&d[2], None).unwrap(),
            &tridiag_eigenvector_for_value_detailed_hp(&d, &e, &d[2], None, 64, options).unwrap(),
        );
    }
}
