//! Cyclic Jacobi on a matrix scaled to unit maximum entry.
/// Arithmetic identity for retained cyclic Jacobi results.
pub const JACOBI_SEMANTICS: &str = "symmetric-jacobi-working-unit-deflation-v2";
use anyhow::{anyhow, Result};
use rug::{ops::Pow, Float};

/// Diagnostics from the independent cyclic Jacobi eigenvalue solver.
#[derive(Clone, Debug)]
pub struct JacobiEigenvaluesHp {
    pub eigenvalues: Vec<Float>,
    pub sweeps: usize,
    pub rotations: usize,
    pub maximum_off_diagonal: Float,
}

/// Computed Jacobi decomposition. Columns of the row-major `eigenvectors`
/// matrix correspond to the ascending `eigenvalues`.
#[derive(Clone, Debug)]
pub struct JacobiEigendecompositionHp {
    pub eigenvalues: Vec<Float>,
    pub eigenvectors: Vec<Float>,
    pub sweeps: usize,
    pub rotations: usize,
    pub maximum_off_diagonal: Float,
}

/// Independent cyclic Jacobi eigenvalues of a finite, exactly symmetric matrix.
///
/// Arithmetic uses `prec` bits; input values are rounded to that precision.
/// Scaling by the largest absolute entry makes the stopping test relative to
/// the matrix rather than to one. The threshold is 2^(-prec) times that scale.
/// This is a computed approximation, not a certified eigenvalue enclosure or
/// a relative-accuracy guarantee for eigenvalues small compared with the norm.
/// A zero budget, invalid domain, unrepresentable arithmetic, or failure to
/// converge returns an error. No Householder or tridiagonal QR code is used.
pub fn dense_symmetric_eigenvalues_jacobi_hp(
    input: &[Float],
    n: usize,
    prec: u32,
    max_sweeps: usize,
) -> Result<JacobiEigenvaluesHp> {
    let result = solve(input, n, prec, max_sweeps, false)?;
    Ok(JacobiEigenvaluesHp {
        eigenvalues: result.eigenvalues,
        sweeps: result.sweeps,
        rotations: result.rotations,
        maximum_off_diagonal: result.maximum_off_diagonal,
    })
}

/// The same computed Jacobi method, accumulating every plane rotation in Q.
/// Repeated eigenvalues have an arbitrary orthonormal basis; a subset cutting
/// a repeated or unresolved cluster does not define a unique spectral subspace.
pub fn dense_symmetric_eigendecomposition_jacobi_hp(
    input: &[Float],
    n: usize,
    prec: u32,
    max_sweeps: usize,
) -> Result<JacobiEigendecompositionHp> {
    solve(input, n, prec, max_sweeps, true)
}

fn maximum_off_diagonal(matrix: &[Float], n: usize, prec: u32) -> Float {
    let mut maximum = Float::with_val(prec, 0);
    for i in 0..n {
        for j in 0..i {
            maximum.max_mut(&matrix[i * n + j].clone().abs());
        }
    }
    maximum
}

fn solve(
    input: &[Float],
    n: usize,
    prec: u32,
    max_sweeps: usize,
    vectors: bool,
) -> Result<JacobiEigendecompositionHp> {
    super::validate_hp_eigen_precision(prec)?;
    if n == 0
        || n.checked_mul(n) != Some(input.len())
        || max_sweeps == 0
        || input.iter().any(|x| !x.is_finite())
        || (0..n).any(|i| (0..i).any(|j| input[i * n + j] != input[j * n + i]))
    {
        return Err(anyhow!("Jacobi requires a finite, exactly symmetric nonempty square matrix and a positive sweep budget"));
    }
    let mut matrix: Vec<_> = input.iter().map(|x| Float::with_val(prec, x)).collect();
    if matrix.iter().any(|x| !x.is_finite()) {
        return Err(anyhow!("Jacobi precision conversion overflowed"));
    }
    let mut scale = matrix
        .iter()
        .map(|x| x.clone().abs())
        .max_by(Float::total_cmp)
        .unwrap();
    if scale.is_zero() {
        scale = Float::with_val(prec, 1);
    }
    for entry in &mut matrix {
        let nonzero = !entry.is_zero();
        *entry /= &scale;
        if !entry.is_finite() || (nonzero && entry.is_zero()) {
            return Err(anyhow!("Jacobi scaling is unrepresentable"));
        }
    }
    let mut basis = if vectors {
        vec![Float::with_val(prec, 0); n * n]
    } else {
        Vec::new()
    };
    if vectors {
        for i in 0..n {
            basis[i * n + i] = Float::with_val(prec, 1);
        }
    }
    let tolerance = Float::with_val(prec, 2).pow(-(prec as i32));
    let mut rotations = 0usize;
    for sweep in 0..=max_sweeps {
        if matrix.iter().chain(&basis).any(|x| !x.is_finite()) {
            return Err(anyhow!("nonfinite Jacobi rotation"));
        }
        let maximum = maximum_off_diagonal(&matrix, n, prec);
        if maximum <= tolerance {
            let mut indices: Vec<_> = (0..n).collect();
            indices.sort_by(|&i, &j| matrix[i * n + i].total_cmp(&matrix[j * n + j]));
            let eigenvalues: Vec<_> = indices
                .iter()
                .map(|&i| Float::with_val(prec, &matrix[i * n + i] * &scale))
                .collect();
            let maximum_off_diagonal = Float::with_val(prec, &maximum * &scale);
            if eigenvalues.iter().any(|x| !x.is_finite())
                || !maximum_off_diagonal.is_finite()
                || indices
                    .iter()
                    .zip(&eigenvalues)
                    .any(|(&i, v)| !matrix[i * n + i].is_zero() && v.is_zero())
                || (!maximum.is_zero() && maximum_off_diagonal.is_zero())
            {
                return Err(anyhow!("Jacobi output rescaling is unrepresentable"));
            }
            let eigenvectors = if vectors {
                (0..n)
                    .flat_map(|i| indices.iter().map(move |&j| (i, j)))
                    .map(|(i, j)| basis[i * n + j].clone())
                    .collect()
            } else {
                Vec::new()
            };
            return Ok(JacobiEigendecompositionHp {
                eigenvalues,
                eigenvectors,
                sweeps: sweep,
                rotations,
                maximum_off_diagonal,
            });
        }
        if sweep == max_sweeps {
            break;
        }
        for p in 0..n - 1 {
            for q in p + 1..n {
                let apq = matrix[p * n + q].clone();
                if apq.clone().abs() <= tolerance {
                    continue;
                }
                let app = matrix[p * n + p].clone();
                let aqq = matrix[q * n + q].clone();
                // t=sign(tau)/(abs(tau)+sqrt(1+tau^2)), tau=(aqq-app)/(2apq).
                // This equivalent form avoids both a huge tau and tau^2.
                let mut delta = Float::with_val(prec, &aqq - &app);
                delta /= 2;
                let tangent = if delta.is_zero() {
                    Float::with_val(prec, 1)
                } else {
                    let mut denominator = delta.clone().hypot(&apq);
                    denominator += delta.clone().abs();
                    let mut t = Float::with_val(prec, &apq / denominator);
                    if delta.is_sign_negative() {
                        t = -t;
                    }
                    t
                };
                let cosine = Float::with_val(prec, 1)
                    / Float::with_val(prec, 1 + Float::with_val(prec, &tangent * &tangent)).sqrt();
                let sine = Float::with_val(prec, &tangent * &cosine);
                for k in 0..n {
                    if k != p && k != q {
                        let akp = matrix[k * n + p].clone();
                        let akq = matrix[k * n + q].clone();
                        let new_p = Float::with_val(
                            prec,
                            Float::with_val(prec, &cosine * &akp)
                                - Float::with_val(prec, &sine * &akq),
                        );
                        let new_q = Float::with_val(
                            prec,
                            Float::with_val(prec, &sine * &akp)
                                + Float::with_val(prec, &cosine * &akq),
                        );
                        matrix[k * n + p] = new_p.clone();
                        matrix[p * n + k] = new_p;
                        matrix[k * n + q] = new_q.clone();
                        matrix[q * n + k] = new_q;
                    }
                    if vectors {
                        let bkp = basis[k * n + p].clone();
                        let bkq = basis[k * n + q].clone();
                        basis[k * n + p] = Float::with_val(
                            prec,
                            Float::with_val(prec, &cosine * &bkp)
                                - Float::with_val(prec, &sine * &bkq),
                        );
                        basis[k * n + q] = Float::with_val(
                            prec,
                            Float::with_val(prec, &sine * &bkp)
                                + Float::with_val(prec, &cosine * &bkq),
                        );
                    }
                }
                let change = Float::with_val(prec, tangent * &apq);
                matrix[p * n + p] = Float::with_val(prec, app - &change);
                matrix[q * n + q] = Float::with_val(prec, aqq + change);
                matrix[p * n + q] = Float::with_val(prec, 0);
                matrix[q * n + p] = Float::with_val(prec, 0);
                rotations = rotations
                    .checked_add(1)
                    .ok_or_else(|| anyhow!("Jacobi rotation counter overflow"))?;
            }
        }
    }
    Err(anyhow!(
        "cyclic Jacobi failed to converge after {max_sweeps} sweeps"
    ))
}
