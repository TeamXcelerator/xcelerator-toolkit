//! Computed centered-flux Dirichlet discretization, not a continuum certificate.
use anyhow::{ensure, Result};

fn grid(n_grid: usize) -> Result<usize> {
    ensure!(
        n_grid > 0 && n_grid < u32::MAX as usize,
        "prolate grid must be positive and below u32::MAX"
    );
    let n = if n_grid.is_multiple_of(2) {
        n_grid + 1
    } else {
        n_grid
    };
    ensure!(
        n < u32::MAX as usize,
        "prolate odd grid plus one must fit u32"
    );
    Ok(n)
}

fn flux(q: usize, position: i64) -> (u128, u128) {
    let square = (q as u128) * (q as u128);
    let lower = position - 1;
    let upper = position + 1;
    (
        square - u128::from(lower.unsigned_abs()).pow(2),
        square - u128::from(upper.unsigned_abs()).pow(2),
    )
}

pub(super) fn build_f64(cfg: &super::ProlateConfig) -> Result<(Vec<f64>, Vec<f64>)> {
    ensure!(
        cfg.lambda.is_finite() && cfg.lambda > 0.0,
        "prolate lambda must be finite and positive"
    );
    ensure!(
        cfg.precision_bits == 53,
        "native prolate arithmetic requires precision_bits=53"
    );
    let n = grid(cfg.n_grid)?;
    ensure!(
        n == cfg.n_grid,
        "native prolate config must contain an odd grid; use ProlateConfig::new"
    );
    let q = n + 1;
    let mut diag = Vec::with_capacity(n);
    let mut off = Vec::with_capacity(n - 1);
    let c = 2.0 * std::f64::consts::PI * cfg.lambda * cfg.lambda;
    for i in 0..n {
        let position = 2 * (i as i64) + 1 - (n as i64);
        let (minus, plus) = flux(q, position);
        let potential = if position == 0 {
            0.0
        } else {
            (c * ((position as f64) / (q as f64))).powi(2)
        };
        let value = ((minus + plus) as f64) / 4.0 + potential;
        ensure!(
            value.is_finite(),
            "prolate matrix entry is outside binary64 range"
        );
        diag.push(value);
        if i > 0 {
            off.push(-(minus as f64) / 4.0);
        }
    }
    Ok((diag, off))
}

#[cfg(feature = "hp")]
pub(super) fn build_hp(
    lambda: &rug::Float,
    n_grid: usize,
    prec: u32,
) -> Result<(Vec<rug::Float>, Vec<rug::Float>)> {
    use rug::Float;
    ensure!(
        (32..=1_000_000).contains(&prec),
        "prolate precision must be in 32..=1000000 bits"
    );
    ensure!(
        lambda.is_finite() && lambda > &0,
        "prolate lambda must be finite and positive"
    );
    let n = grid(n_grid)?;
    let q = n + 1;
    let mut c = Float::with_val(prec, rug::float::Constant::Pi);
    c *= 2u32;
    c *= lambda;
    c *= lambda;
    let mut diag = Vec::with_capacity(n);
    let mut off = Vec::with_capacity(n - 1);
    for i in 0..n {
        let position = 2 * (i as i64) + 1 - (n as i64);
        let (minus, plus) = flux(q, position);
        let mut value = Float::with_val(prec, minus + plus);
        value /= 4u32;
        if position != 0 {
            let mut potential = Float::with_val(prec, position);
            potential /= q;
            potential *= &c;
            potential.square_mut();
            value += potential;
        }
        ensure!(
            value.is_finite(),
            "prolate matrix entry is outside MPFR range"
        );
        diag.push(value);
        if i > 0 {
            let mut coefficient = Float::with_val(prec, minus);
            coefficient /= -4i32;
            off.push(coefficient);
        }
    }
    Ok((diag, off))
}

#[cfg(test)]
mod tests {
    #[test]
    fn native_parity_includes_the_center_coordinate() {
        use crate::prolate::{parity_of_f64, Parity};
        for scale in [1e-200, 1.0, 1e200] {
            assert_eq!(
                parity_of_f64(&[scale, scale, -scale]),
                Parity::Indeterminate
            );
            assert_eq!(parity_of_f64(&[0.0, scale, 0.0]), Parity::Even);
            assert_eq!(parity_of_f64(&[scale, 0.0, -scale]), Parity::Odd);
        }
    }
}
