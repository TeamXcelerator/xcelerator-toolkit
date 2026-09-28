//! Real-part crossing candidates. A crossing is not a complex zero, and point
//! sampling does not establish completeness, continuity, or certified isolation.
use anyhow::{bail, Result};
use rayon::prelude::*;

/// Legacy name for real-part crossing candidates only. The imaginary part
/// need not vanish. Invalid grids/callbacks panic; use the checked API.
pub fn scan_critical_line_zeros_f64<F>(
    eval_fn: &F,
    t_min: f64,
    t_max: f64,
    n_scan: usize,
) -> Vec<f64>
where
    F: Fn(f64, f64) -> (f64, f64) + Sync,
{
    try_scan_critical_line_real_crossings_f64(eval_fn, t_min, t_max, n_scan)
        .expect("valid finite real-part crossing scan required")
}

/// Return sampled real-part zeros and bracketed real-part crossing candidates.
/// Bisection is bounded to 50 steps; no tolerance or complex-zero certificate
/// is implied. Continuity and numerical accuracy are caller obligations.
pub fn try_scan_critical_line_real_crossings_f64<F>(
    eval_fn: &F,
    t_min: f64,
    t_max: f64,
    n_scan: usize,
) -> Result<Vec<f64>>
where
    F: Fn(f64, f64) -> (f64, f64) + Sync,
{
    if !t_min.is_finite()
        || !t_max.is_finite()
        || t_min >= t_max
        || n_scan == 0
        || n_scan.checked_add(1).is_none()
        || (n_scan as u128) > (1u128 << 53)
    {
        bail!("crossing scan requires finite increasing bounds and a positive representable grid size");
    }
    let evaluate = |t| -> Result<f64> {
        let (re, im) = eval_fn(0.5, t);
        if !re.is_finite() || !im.is_finite() {
            bail!("nonfinite crossing evaluation");
        }
        Ok(re)
    };
    let grid: Vec<f64> = (0..=n_scan)
        .map(|i| {
            if i == 0 {
                return t_min;
            }
            if i == n_scan {
                return t_max;
            }
            let fraction = i as f64 / n_scan as f64;
            if t_min.is_sign_negative() == t_max.is_sign_negative() {
                t_min + (t_max - t_min) * fraction
            } else {
                t_min * (1.0 - fraction) + t_max * fraction
            }
        })
        .collect();
    if grid.iter().any(|v| !v.is_finite()) || grid.windows(2).any(|w| w[0] >= w[1]) {
        bail!("crossing grid collapses at binary64 precision");
    }
    let values = grid
        .par_iter()
        .map(|&t| evaluate(t))
        .collect::<Result<Vec<_>>>()?;
    let mut roots = Vec::new();
    for i in 0..=n_scan {
        if values[i] == 0.0 {
            roots.push(grid[i]);
            continue;
        }
        if i == 0
            || values[i - 1] == 0.0
            || values[i - 1].is_sign_negative() == values[i].is_sign_negative()
        {
            continue;
        }
        let (mut a, mut b, mut fa) = (grid[i - 1], grid[i], values[i - 1]);
        let mut exact = None;
        for _ in 0..50 {
            let middle = a.midpoint(b);
            let fm = evaluate(middle)?;
            if fm == 0.0 {
                exact = Some(middle);
                break;
            }
            if middle == a || middle == b {
                break;
            }
            if fa.is_sign_negative() != fm.is_sign_negative() {
                b = middle;
            } else {
                a = middle;
                fa = fm;
            }
        }
        roots.push(exact.unwrap_or_else(|| a.midpoint(b)));
    }
    Ok(roots)
}

/// Legacy HP name for real-part crossings, not complex zeros. Invalid input
/// panics; use the checked API to propagate an error.
#[cfg(feature = "hp")]
pub fn scan_critical_line_zeros_hp<F>(
    eval_fn: &F,
    t_min: &rug::Float,
    t_max: &rug::Float,
    n_scan: usize,
    bisect_iter: usize,
) -> Vec<rug::Float>
where
    F: Fn(&rug::Float, &rug::Float) -> (rug::Float, rug::Float) + Sync,
{
    try_scan_critical_line_real_crossings_hp(eval_fn, t_min, t_max, n_scan, bisect_iter)
        .expect("valid finite HP real-part crossing scan required")
}

/// Checked HP real-part crossing candidates. Callback precision must be at
/// least the grid precision. Exact sampled zeros are included once; bounded
/// bisection does not certify tolerance, isolation, or a complex zero.
#[cfg(feature = "hp")]
pub fn try_scan_critical_line_real_crossings_hp<F>(
    eval_fn: &F,
    t_min: &rug::Float,
    t_max: &rug::Float,
    n_scan: usize,
    bisect_iter: usize,
) -> Result<Vec<rug::Float>>
where
    F: Fn(&rug::Float, &rug::Float) -> (rug::Float, rug::Float) + Sync,
{
    xc_numerics::hp_runtime::run_hp(|| scan_hp(eval_fn, t_min, t_max, n_scan, bisect_iter))
}

#[cfg(feature = "hp")]
fn scan_hp<F>(
    eval_fn: &F,
    t_min: &rug::Float,
    t_max: &rug::Float,
    n_scan: usize,
    bisect_iter: usize,
) -> Result<Vec<rug::Float>>
where
    F: Fn(&rug::Float, &rug::Float) -> (rug::Float, rug::Float) + Sync,
{
    use rug::Float;
    use xc_numerics::mpfr_interval::MpfrInterval;
    let prec = t_min.prec().max(t_max.prec());
    if !(32..=1_000_000).contains(&prec)
        || !t_min.is_finite()
        || !t_max.is_finite()
        || t_min >= t_max
        || n_scan == 0
        || n_scan.checked_add(1).is_none()
    {
        bail!("invalid HP crossing grid domain or precision");
    }
    let sigma = Float::with_val(prec, 0.5);
    let evaluate = |t: &Float| -> Result<Float> {
        let (re, im) = eval_fn(&sigma, t);
        if !re.is_finite() || !im.is_finite() || re.prec() < prec || im.prec() < prec {
            bail!("crossing callback returned a nonfinite or lower-precision value");
        }
        Ok(re)
    };
    let grid: Vec<Float> = (0..=n_scan)
        .map(|i| {
            if i == 0 {
                return Float::with_val(prec, t_min);
            }
            if i == n_scan {
                return Float::with_val(prec, t_max);
            }
            let mut fraction = Float::with_val(prec, i);
            fraction /= n_scan;
            if t_min.is_sign_negative() == t_max.is_sign_negative() {
                let mut value = Float::with_val(prec, t_max - t_min);
                value *= fraction;
                value += t_min;
                value
            } else {
                let mut complement = Float::with_val(prec, 1);
                complement -= &fraction;
                let mut value = Float::with_val(prec, t_min * complement);
                value += Float::with_val(prec, t_max * fraction);
                value
            }
        })
        .collect();
    if grid.iter().any(|v| !v.is_finite()) || grid.windows(2).any(|w| w[0] >= w[1]) {
        bail!("HP crossing grid is unrepresentable at working precision");
    }
    let values = grid.par_iter().map(evaluate).collect::<Result<Vec<_>>>()?;
    let midpoint = |a: &Float, b: &Float| -> Result<Float> {
        Ok(MpfrInterval::new(a.clone(), b.clone())?
            .midpoint_point()
            .lower()
            .clone())
    };
    let mut roots = Vec::new();
    for i in 0..=n_scan {
        if values[i].is_zero() {
            roots.push(grid[i].clone());
            continue;
        }
        if i == 0
            || values[i - 1].is_zero()
            || values[i - 1].is_sign_negative() == values[i].is_sign_negative()
        {
            continue;
        }
        let (mut a, mut b, mut fa) = (grid[i - 1].clone(), grid[i].clone(), values[i - 1].clone());
        let mut exact = None;
        for _ in 0..bisect_iter {
            let middle = midpoint(&a, &b)?;
            let fm = evaluate(&middle)?;
            if fm.is_zero() {
                exact = Some(middle);
                break;
            }
            if middle == a || middle == b {
                break;
            }
            if fa.is_sign_negative() != fm.is_sign_negative() {
                b = middle;
            } else {
                a = middle;
                fa = fm;
            }
        }
        roots.push(match exact {
            Some(root) => root,
            None => midpoint(&a, &b)?,
        });
    }
    Ok(roots)
}
