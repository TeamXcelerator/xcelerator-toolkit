//! Checked pointwise reconstruction and discrete least-squares comparison.
use anyhow::{ensure, Result};

pub(super) fn compare_f64(
    xi: &[f64],
    n: usize,
    lambda: f64,
    u: &[f64],
    k: &[f64],
) -> Result<super::ComparisonResult> {
    crate::distance::validate_even_basis_f64(xi, n, lambda)?;
    ensure!(
        !u.is_empty() && u.len() == k.len(),
        "comparison grids must have the same positive length"
    );
    ensure!(
        u.iter().all(|v| v.is_finite() && *v > 0.0) && k.iter().all(|v| v.is_finite()),
        "comparison samples must be finite and abscissae positive"
    );
    let l = crate::distance::log_length_f64(lambda);
    let inverse = 1.0 / l.sqrt();
    let x: Vec<f64> = u
        .iter()
        .map(|value| {
            let phase =
                2.0 * std::f64::consts::PI * crate::distance::log_product_f64(lambda, *value) / l;
            let mut sum = xi[n];
            for j in 1..=n {
                sum += 2.0 * xi[n + j] * ((j as f64) * phase).cos();
            }
            sum * inverse
        })
        .collect();
    ensure!(
        x.iter().all(|v| v.is_finite()),
        "reconstructed comparison values are nonfinite"
    );
    let sx = x.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    let sk = k.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    ensure!(sk > 0.0, "comparison candidate is the zero vector");
    let mut c = 0.0;
    if sx > 0.0 {
        let cross: f64 = x.iter().zip(k).map(|(x, k)| (x / sx) * (k / sk)).sum();
        let square: f64 = k.iter().map(|k| (k / sk).powi(2)).sum();
        let q = cross / square;
        c = (q * sx) / sk;
        if !c.is_finite() || (c == 0.0 && q != 0.0) {
            c = (sx / sk) * q;
        }
        ensure!(
            c.is_finite() && (c != 0.0 || q == 0.0),
            "optimal comparison scalar is outside binary64 range"
        );
    }
    let mut error = 0.0f64;
    let mut index = 0;
    let mut l2 = 0.0f64;
    let mut xl2 = 0.0f64;
    for (i, (x, k)) in x.iter().zip(k).enumerate() {
        let residual = x - c * k;
        ensure!(residual.is_finite(), "comparison residual is nonfinite");
        if residual.abs() > error {
            error = residual.abs();
            index = i;
        }
        l2 = l2.hypot(residual);
        xl2 = xl2.hypot(*x);
    }
    ensure!(
        l2.is_finite() && xl2.is_finite(),
        "comparison norm is outside binary64 range"
    );
    Ok(super::ComparisonResult {
        optimal_scalar: c,
        linf_error: error,
        l2_error: l2,
        xi_linf: sx,
        xi_l2: xl2,
        linf_index: index,
    })
}

#[cfg(feature = "hp")]
pub(super) fn compare_hp(
    xi: &[rug::Float],
    n: usize,
    lambda: &rug::Float,
    u: &[rug::Float],
    k: &[rug::Float],
    p: u32,
) -> Result<super::hp::HpComparisonResult> {
    use rug::Float;
    crate::distance::hp::validate_even_basis(xi, n, lambda, p)?;
    ensure!(
        !u.is_empty() && u.len() == k.len(),
        "comparison grids must have the same positive length"
    );
    ensure!(
        u.iter().all(|v| v.is_finite() && v > &0) && k.iter().all(Float::is_finite),
        "comparison samples must be finite and abscissae positive"
    );
    let l = crate::distance::hp::log_length(lambda, p);
    let mut inverse = Float::with_val(p, 1);
    inverse /= l.clone().sqrt();
    let two_pi = Float::with_val(p, rug::float::Constant::Pi) * 2u32;
    let x: Vec<Float> = u
        .iter()
        .map(|u| {
            let mut phase = crate::distance::hp::log_product(lambda, u, p);
            phase *= &two_pi;
            phase /= &l;
            let mut sum = Float::with_val(p, &xi[n]);
            for j in 1..=n {
                let mut angle = phase.clone();
                angle *= j;
                angle.cos_mut();
                angle *= 2u32;
                angle *= &xi[n + j];
                sum += angle;
            }
            sum * &inverse
        })
        .collect();
    ensure!(
        x.iter().all(Float::is_finite),
        "reconstructed comparison values are nonfinite"
    );
    let maximum = |v: &[Float]| {
        v.iter()
            .map(|v| Float::with_val(p, v).abs())
            .max_by(|a, b| a.partial_cmp(b).expect("finite samples"))
            .unwrap()
    };
    let sx = maximum(&x);
    let sk = maximum(k);
    ensure!(!sk.is_zero(), "comparison candidate is the zero vector");
    let mut c = Float::with_val(p, 0);
    if !sx.is_zero() {
        let cross: Vec<Float> = x
            .iter()
            .zip(k)
            .map(|(x, k)| {
                let mut term = Float::with_val(p, x / &sx);
                term *= Float::with_val(p, k / &sk);
                term
            })
            .collect();
        let squares: Vec<Float> = k
            .iter()
            .map(|k| Float::with_val(p, k / &sk).square())
            .collect();
        let mut q = xc_numerics::reduction::deterministic_pairwise_sum_hp(&cross, p);
        q /= xc_numerics::reduction::deterministic_pairwise_sum_hp(&squares, p);
        c = Float::with_val(p, &q * &sx);
        c /= &sk;
        if !c.is_finite() || (c.is_zero() && !q.is_zero()) {
            c = Float::with_val(p, &sx / &sk);
            c *= &q;
        }
        ensure!(
            c.is_finite() && (!c.is_zero() || q.is_zero()),
            "optimal comparison scalar is outside MPFR range"
        );
    }
    let mut error = Float::with_val(p, 0);
    let mut index = 0;
    let mut residuals = Vec::with_capacity(x.len());
    for (i, (x, k)) in x.iter().zip(k).enumerate() {
        let mut residual = Float::with_val(p, &c * k);
        residual = -residual;
        residual += x;
        ensure!(residual.is_finite(), "comparison residual is nonfinite");
        if residual.clone().abs() > error {
            error = residual.clone().abs();
            index = i;
        }
        residuals.push(residual);
    }
    let norm = |values: &[Float]| {
        let scale = maximum(values);
        if scale.is_zero() {
            return scale;
        }
        let terms: Vec<Float> = values
            .iter()
            .map(|v| Float::with_val(p, v / &scale).square())
            .collect();
        let mut result = xc_numerics::reduction::deterministic_pairwise_sum_hp(&terms, p).sqrt();
        result *= scale;
        result
    };
    let l2 = norm(&residuals);
    let xl2 = norm(&x);
    ensure!(
        l2.is_finite() && xl2.is_finite(),
        "comparison norm is outside MPFR range"
    );
    Ok(super::hp::HpComparisonResult {
        optimal_scalar: c,
        linf_error: error,
        l2_error: l2,
        xi_linf: sx,
        xi_l2: xl2,
        linf_index: index,
    })
}
