// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Completion of approximate binary64 symmetric eigensystems.
//!
//! A library QR decomposition can return an orthogonal basis whose columns
//! are paired with the wrong eigenvalues, or whose 2x2 deflations lose
//! eigenvector accuracy through cancellation. Given any orthogonal basis `Q`,
//! `B = Q^T A Q` is an orthogonal similarity of `A`. Cyclic Jacobi rotations
//! diagonalize `B`, so the diagonal pairs every returned column with its own
//! Rayleigh quotient, and the residual is verified against `A` itself.

use anyhow::{bail, Result};

const MAXIMUM_SWEEPS: usize = 64;

/// Complete a symmetric eigensystem from an approximately orthogonal basis.
///
/// `matrix` and `basis` are column-major `n x n`; only the lower triangle of
/// `matrix` is read, as a symmetric QR routine does. On success `basis` holds
/// orthonormal eigenvectors and the returned eigenvalues are ascending, each
/// paired with its column. Every pair must satisfy
/// `|A v - lambda v| <= 64 (n + 1) eps |A|_F`; otherwise an error is returned.
pub fn complete_symmetric_eigensystem_f64(
    matrix: &[f64],
    n: usize,
    basis: &mut [f64],
) -> Result<Vec<f64>> {
    let size = n.checked_mul(n);
    if n == 0
        || size != Some(matrix.len())
        || size != Some(basis.len())
        || basis.iter().any(|v| !v.is_finite())
        || (0..n).any(|column| (column..n).any(|row| !matrix[column * n + row].is_finite()))
    {
        bail!("symmetric eigensystem completion requires finite nonempty square inputs");
    }
    // Exact power-of-two scaling keeps every product in range.
    let maximum = (0..n)
        .flat_map(|column| (column..n).map(move |row| matrix[column * n + row].abs()))
        .fold(0.0_f64, f64::max);
    let exponent = if maximum == 0.0 {
        0
    } else {
        (((maximum.to_bits() >> 52) & 0x7ff) as i32 - 1023).max(-1022)
    };
    let scale = 2.0_f64.powi(exponent);
    let a = |row: usize, column: usize| {
        let (row, column) = if row >= column {
            (row, column)
        } else {
            (column, row)
        };
        matrix[column * n + row] / scale
    };
    let frobenius = (0..n)
        .flat_map(|row| (0..n).map(move |column| (row, column)))
        .map(|(row, column)| a(row, column).powi(2))
        .sum::<f64>()
        .sqrt();

    // B = Q^T (A Q), row-major and exactly symmetric.
    let mut product = vec![0.0_f64; n * n];
    for column in 0..n {
        for row in 0..n {
            product[column * n + row] = (0..n)
                .map(|k| a(row, k) * basis[column * n + k])
                .sum::<f64>();
        }
    }
    let mut b = vec![0.0_f64; n * n];
    for p in 0..n {
        for q in 0..=p {
            let left = (0..n)
                .map(|k| basis[p * n + k] * product[q * n + k])
                .sum::<f64>();
            let right = (0..n)
                .map(|k| basis[q * n + k] * product[p * n + k])
                .sum::<f64>();
            let value = 0.5 * (left + right);
            b[p * n + q] = value;
            b[q * n + p] = value;
        }
    }

    let mut converged = false;
    for _ in 0..MAXIMUM_SWEEPS {
        let mut rotated = false;
        for p in 0..n {
            for q in p + 1..n {
                let bpq = b[p * n + q];
                let (bpp, bqq) = (b[p * n + p], b[q * n + q]);
                if bpq.abs() <= f64::EPSILON * (bpp.abs() * bqq.abs()).sqrt()
                    || bpq.abs() <= f64::EPSILON * f64::EPSILON * frobenius
                {
                    continue;
                }
                rotated = true;
                let theta = (bqq - bpp) / (2.0 * bpq);
                let t = theta.signum() / (theta.abs() + theta.hypot(1.0));
                let c = 1.0 / t.hypot(1.0);
                let s = t * c;
                for k in 0..n {
                    let (kp, kq) = (b[k * n + p], b[k * n + q]);
                    b[k * n + p] = c * kp - s * kq;
                    b[k * n + q] = s * kp + c * kq;
                }
                for k in 0..n {
                    let (pk, qk) = (b[p * n + k], b[q * n + k]);
                    b[p * n + k] = c * pk - s * qk;
                    b[q * n + k] = s * pk + c * qk;
                }
                b[p * n + p] = bpp - t * bpq;
                b[q * n + q] = bqq + t * bpq;
                b[p * n + q] = 0.0;
                b[q * n + p] = 0.0;
                for k in 0..n {
                    let (vp, vq) = (basis[p * n + k], basis[q * n + k]);
                    basis[p * n + k] = c * vp - s * vq;
                    basis[q * n + k] = s * vp + c * vq;
                }
            }
        }
        if !rotated {
            converged = true;
            break;
        }
    }
    if !converged {
        bail!("symmetric eigensystem completion did not converge");
    }

    let mut order = (0..n).collect::<Vec<_>>();
    order.sort_by(|&x, &y| b[x * n + x].total_cmp(&b[y * n + y]));
    let original = basis.to_vec();
    let mut values = Vec::with_capacity(n);
    for (target, &source) in order.iter().enumerate() {
        basis[target * n..(target + 1) * n]
            .copy_from_slice(&original[source * n..(source + 1) * n]);
        values.push(b[source * n + source]);
    }

    // Verify every pair against A itself, not against the similarity.
    let tolerance = 64.0 * (n as f64 + 1.0) * f64::EPSILON * frobenius.max(f64::MIN_POSITIVE);
    for (column, &value) in values.iter().enumerate() {
        let vector = &basis[column * n..(column + 1) * n];
        let residual = (0..n)
            .map(|row| (0..n).map(|k| a(row, k) * vector[k]).sum::<f64>() - value * vector[row])
            .map(|r| r * r)
            .sum::<f64>()
            .sqrt();
        let norm = vector.iter().map(|v| v * v).sum::<f64>().sqrt();
        if !residual.is_finite()
            || residual > tolerance
            || (norm - 1.0).abs() > 64.0 * (n as f64 + 1.0) * f64::EPSILON
        {
            bail!("symmetric eigensystem completion missed binary64 backward accuracy");
        }
    }
    for column in 0..n {
        for other in 0..column {
            let overlap = (0..n)
                .map(|row| basis[column * n + row] * basis[other * n + row])
                .sum::<f64>();
            if !overlap.is_finite() || overlap.abs() > 64.0 * (n as f64 + 1.0) * f64::EPSILON {
                bail!("symmetric eigensystem completion requires an orthogonal basis");
            }
        }
    }
    let scaled_values = values.iter().map(|v| v * scale).collect::<Vec<_>>();
    if values
        .iter()
        .zip(&scaled_values)
        .any(|(before, after)| *before != 0.0 && *after == 0.0)
    {
        bail!("symmetric eigenvalue rescaling underflowed binary64");
    }
    let values = scaled_values;
    if values.iter().any(|v| !v.is_finite()) {
        bail!("symmetric eigenvalues are outside binary64 range");
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_eigenvalues_do_not_admit_duplicate_basis_columns() {
        let matrix = [0.0; 4];
        let mut repeated = [1.0, 0.0, 1.0, 0.0];
        assert!(complete_symmetric_eigensystem_f64(&matrix, 2, &mut repeated).is_err());
        let mut orthogonal = [1.0, 0.0, 0.0, 1.0];
        assert_eq!(
            complete_symmetric_eigensystem_f64(&matrix, 2, &mut orthogonal).unwrap(),
            vec![0.0, 0.0]
        );
    }

    #[test]
    fn eigenvalues_are_paired_with_their_own_columns() {
        // Two tiny eigenvalues beside a unit one. A basis whose columns are
        // exact eigenvectors, listed out of order, must be relabeled.
        let matrix = [1.0, 0.0, 0.0, 0.0, 1e-10, 1e-20, 0.0, 1e-20, 2e-10];
        let mut basis = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let values = complete_symmetric_eigensystem_f64(&matrix, 3, &mut basis).unwrap();
        assert!((values[0] - 1e-10).abs() < 1e-24 && (values[1] - 2e-10).abs() < 1e-24);
        assert_eq!(values[2], 1.0);
        assert!((basis[1] - 1.0).abs() < 1e-9 && (basis[5] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn a_coarse_orthogonal_basis_is_completed_to_backward_accuracy() {
        // [[-1, 5e-8], [5e-8, 3]] from the identity basis: the eigenvector
        // tilt 1.25e-8 must be resolved, not left at the axis directions.
        let matrix = [-1.0, 5e-8, 5e-8, 3.0];
        let mut basis = [1.0, 0.0, 0.0, 1.0];
        let values = complete_symmetric_eigensystem_f64(&matrix, 2, &mut basis).unwrap();
        let shift = 6.25e-16;
        assert!((values[0] - (-1.0 - shift)).abs() <= 4.0 * f64::EPSILON);
        assert!((values[1] - (3.0 + shift)).abs() <= 8.0 * f64::EPSILON);
        let tilt = basis[1] / basis[0];
        assert!((tilt + 1.25e-8).abs() < 1e-22, "{tilt:e}");
        assert!(complete_symmetric_eigensystem_f64(&[f64::NAN], 1, &mut [1.0]).is_err());
        assert!(complete_symmetric_eigensystem_f64(&[1.0], 1, &mut [0.5]).is_err());
    }
}
