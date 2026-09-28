//! Computed residual gates for eigenvector recovery, independent of the solve.
//! Coefficients are scaled before multiplication to avoid exponent overflow.
use anyhow::{anyhow, Result};
use rug::Float;

fn setup<'a>(
    entries: impl Iterator<Item = &'a Float>,
    lambda: &Float,
    v: &[Float],
    p: u32,
) -> Result<(Float, u32, Float)> {
    if v.is_empty()
        || v.iter().any(|x| !x.is_finite())
        || v.iter().all(Float::is_zero)
        || !lambda.is_finite()
    {
        return Err(anyhow!("invalid recovered eigenvector or eigenvalue"));
    }
    let mut bits = p.max(lambda.prec());
    let mut scale = lambda.clone().abs();
    for value in entries {
        if !value.is_finite() {
            return Err(anyhow!("nonfinite eigenvector source matrix"));
        }
        bits = bits.max(value.prec());
        if value.clone().abs() > scale {
            scale = value.clone().abs();
        }
    }
    bits = bits
        .checked_add(32)
        .filter(|bits| *bits <= rug::float::prec_max())
        .ok_or_else(|| anyhow!("eigenvector residual guard precision is unsupported"))?;
    if scale.is_zero() {
        scale = Float::with_val(bits, 1);
    }
    let mut tolerance = Float::with_val(bits, 1) >> (p / 2);
    tolerance *= v.len();
    Ok((scale, bits, tolerance))
}

fn finish(residual: Float, tolerance: Float) -> Result<()> {
    if !residual.is_finite() || residual > tolerance {
        return Err(anyhow!("eigenvector recovery did not meet the computed scaled residual target: residual={residual}, target={tolerance}"));
    }
    Ok(())
}

pub(super) fn tridiagonal(
    d: &[Float],
    e: &[Float],
    lambda: &Float,
    v: &[Float],
    p: u32,
    shift_error: Option<&Float>,
) -> Result<()> {
    let (scale, bits, mut tolerance) = setup(d.iter().chain(e), lambda, v, p)?;
    if let Some(error) = shift_error {
        if !error.is_finite() || error < &0 {
            return Err(anyhow!("invalid selected shift uncertainty"));
        }
        let maximum_component = v
            .iter()
            .map(|x| x.clone().abs())
            .max_by(Float::total_cmp)
            .unwrap();
        tolerance += Float::with_val(bits, error / &scale) * maximum_component;
        if !tolerance.is_finite() {
            return Err(anyhow!("unrepresentable selected shift uncertainty"));
        }
    }
    let normalized_lambda = Float::with_val(bits, lambda) / &scale;
    let mut maximum = Float::with_val(bits, 0);
    for i in 0..d.len() {
        let mut r = Float::with_val(bits, &d[i]) / &scale;
        r -= &normalized_lambda;
        r *= &v[i];
        if i > 0 {
            r += (Float::with_val(bits, &e[i - 1]) / &scale) * &v[i - 1];
        }
        if i + 1 < d.len() {
            r += (Float::with_val(bits, &e[i]) / &scale) * &v[i + 1];
        }
        if !r.is_finite() {
            return Err(anyhow!("nonfinite scaled tridiagonal residual"));
        }
        if r.clone().abs() > maximum {
            maximum = r.abs();
        }
    }
    finish(maximum, tolerance)
}

pub(super) fn dense(a: &[Float], n: usize, lambda: &Float, v: &[Float], p: u32) -> Result<()> {
    let (scale, bits, tolerance) = setup(a.iter(), lambda, v, p)?;
    let normalized_lambda = Float::with_val(bits, lambda) / &scale;
    let mut maximum = Float::with_val(bits, 0);
    for i in 0..n {
        let mut r = Float::with_val(bits, 0);
        for j in 0..n {
            r += (Float::with_val(bits, &a[i * n + j]) / &scale) * &v[j];
        }
        r -= Float::with_val(bits, &normalized_lambda * &v[i]);
        if !r.is_finite() {
            return Err(anyhow!("nonfinite scaled dense residual"));
        }
        if r.clone().abs() > maximum {
            maximum = r.abs();
        }
    }
    finish(maximum, tolerance)
}

/// Whether a recovered vector belongs to the requested eigenvalue. Shifted
/// inverse iteration converges to the eigenvalue nearest its shift, so a
/// neighbor nearer the shift yields the neighbor's exact eigenvector, whose
/// residual is only the eigenvalue separation. Its Rayleigh quotient exposes it.
fn pairs(
    quotient: Float,
    lambda: &Float,
    scale: &Float,
    bits: u32,
    p: u32,
    n: usize,
    shift_error: Option<&Float>,
) -> Result<bool> {
    let mut tolerance = Float::with_val(bits, 1) >> p.saturating_sub(16);
    tolerance *= n;
    if let Some(error) = shift_error {
        tolerance += Float::with_val(bits, error / scale);
    }
    let mut difference = quotient;
    difference -= Float::with_val(bits, lambda / scale);
    if !difference.is_finite() || !tolerance.is_finite() {
        return Err(anyhow!("nonfinite eigenvector pairing check"));
    }
    Ok(difference.abs() <= tolerance)
}

pub(super) fn tridiagonal_pairs(
    d: &[Float],
    e: &[Float],
    lambda: &Float,
    v: &[Float],
    p: u32,
    shift_error: Option<&Float>,
) -> Result<bool> {
    let (scale, bits, _) = setup(d.iter().chain(e), lambda, v, p)?;
    let mut numerator = Float::with_val(bits, 0);
    let mut denominator = Float::with_val(bits, 0);
    for i in 0..d.len() {
        let square = Float::with_val(bits, &v[i] * &v[i]);
        numerator += Float::with_val(bits, &d[i] / &scale) * &square;
        denominator += square;
        if i + 1 < d.len() {
            let mut coupling = Float::with_val(bits, &e[i] / &scale) * &v[i];
            coupling *= &v[i + 1];
            coupling *= 2u32;
            numerator += coupling;
        }
    }
    pairs(
        numerator / denominator,
        lambda,
        &scale,
        bits,
        p,
        d.len(),
        shift_error,
    )
}

pub(super) fn dense_pairs(
    a: &[Float],
    n: usize,
    lambda: &Float,
    v: &[Float],
    p: u32,
) -> Result<bool> {
    let (scale, bits, _) = setup(a.iter(), lambda, v, p)?;
    let mut numerator = Float::with_val(bits, 0);
    let mut denominator = Float::with_val(bits, 0);
    for i in 0..n {
        let mut row = Float::with_val(bits, 0);
        for j in 0..n {
            row += Float::with_val(bits, &a[i * n + j] / &scale) * &v[j];
        }
        numerator += row * &v[i];
        denominator += Float::with_val(bits, &v[i] * &v[i]);
    }
    pairs(numerator / denominator, lambda, &scale, bits, p, n, None)
}
