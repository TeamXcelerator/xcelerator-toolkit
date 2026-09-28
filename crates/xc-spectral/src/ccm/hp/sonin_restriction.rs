//! Computed compression onto a band-concentration spectral subspace.
use anyhow::{anyhow, Result};
use rug::{ops::Pow, Float};

/// Retain the lowest `n - n_drop` concentration modes and compute Q^T A Q.
/// The checks are computed residual screens, not interval certificates.
pub(super) fn restricted_spectrum(
    a: &[Float],
    c: &[Float],
    n: usize,
    n_drop: usize,
    p: u32,
    sweeps: usize,
) -> Result<(Vec<Float>, Vec<Float>)> {
    if n == 0
        || n.checked_mul(n) != Some(a.len())
        || c.len() != a.len()
        || n_drop > n
        || a.iter().any(|x| !x.is_finite())
        || (0..n).any(|i| (0..i).any(|j| a[i * n + j] != a[j * n + i]))
    {
        return Err(anyhow!(
            "invalid Sonin compression matrix or dropped dimension"
        ));
    }
    let decomposition =
        xc_numerics::eigen::dense_symmetric_eigendecomposition_jacobi_hp(c, n, p, sweeps)?;
    let chi = decomposition.eigenvalues;
    let q = decomposition.eigenvectors;
    let keep = n - n_drop;
    let threshold = Float::with_val(p, Float::with_val(p, 2).pow(-((p / 2) as i32)) * n);
    let mut cscale = c
        .iter()
        .map(|x| x.clone().abs())
        .max_by(Float::total_cmp)
        .unwrap();
    if cscale.is_zero() {
        cscale = Float::with_val(p, 1);
    }
    if keep > 0 && keep < n {
        let gap = Float::with_val(p, Float::with_val(p, &chi[keep] - &chi[keep - 1]) / &cscale);
        if !gap.is_finite() || gap <= threshold {
            return Err(anyhow!(
                "Sonin cutoff splits a repeated or unresolved concentration cluster"
            ));
        }
    }
    for i in 0..n {
        for j in 0..n {
            let mut gram = Float::with_val(p, 0);
            let mut residual = Float::with_val(p, 0);
            for k in 0..n {
                gram += Float::with_val(p, &q[k * n + i] * &q[k * n + j]);
                residual += Float::with_val(
                    p,
                    Float::with_val(p, &c[i * n + k] / &cscale) * &q[k * n + j],
                );
            }
            gram -= usize::from(i == j);
            residual -= Float::with_val(p, Float::with_val(p, &chi[j] / &cscale) * &q[i * n + j]);
            if !gram.is_finite()
                || !residual.is_finite()
                || gram.abs() > threshold
                || residual.abs() > threshold
            {
                return Err(anyhow!("Sonin concentration basis failed its computed residual or orthogonality screen"));
            }
        }
    }
    if keep == 0 {
        return Ok((chi, Vec::new()));
    }
    // Matrix-vector products first: O(n^2 keep + n keep^2), rather than O(n^4).
    let mut aq = vec![Float::with_val(p, 0); n * keep];
    for i in 0..n {
        for j in 0..keep {
            for k in 0..n {
                aq[i * keep + j] += Float::with_val(p, &a[i * n + k] * &q[k * n + j]);
            }
        }
    }
    let mut compressed = vec![Float::with_val(p, 0); keep * keep];
    for i in 0..keep {
        for j in i..keep {
            let mut value = Float::with_val(p, 0);
            for k in 0..n {
                value += Float::with_val(p, &q[k * n + i] * &aq[k * keep + j]);
            }
            if !value.is_finite() {
                return Err(anyhow!("Sonin compression arithmetic is unrepresentable"));
            }
            compressed[i * keep + j] = value.clone();
            compressed[j * keep + i] = value;
        }
    }
    let result =
        xc_numerics::eigen::dense_symmetric_eigenvalues_jacobi_hp(&compressed, keep, p, sweeps)?;
    Ok((chi, result.eigenvalues))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sonin_compression_has_the_restricted_sign_when_a_finite_penalty_does_not() {
        let p = 128;
        let a = vec![
            Float::with_val(p, -1),
            Float::with_val(p, 10),
            Float::with_val(p, 10),
            Float::with_val(p, 1) / 100,
        ];
        let c = vec![
            Float::with_val(p, 1),
            Float::with_val(p, 0),
            Float::with_val(p, 0),
            Float::with_val(p, 0),
        ];
        let (_, spectrum) = restricted_spectrum(&a, &c, 2, 1, p, 40).unwrap();
        assert_eq!(spectrum.len(), 1);
        assert_eq!(spectrum[0], a[3]);
        let mut penalty = a.clone();
        penalty[0] += 111;
        let wrong =
            xc_numerics::eigen::dense_symmetric_eigenvalues_jacobi_hp(&penalty, 2, p, 40).unwrap();
        assert!(wrong.eigenvalues[0] < 0);
        assert!(spectrum[0] > 0);
    }
    #[test]
    fn sonin_compression_uses_rotated_complement_and_handles_dimension_endpoints() {
        let p = 128;
        let a = vec![
            Float::with_val(p, 2),
            Float::with_val(p, 3),
            Float::with_val(p, 3),
            Float::with_val(p, 5),
        ];
        let c = vec![Float::with_val(p, 0.5); 4];
        let (_, spectrum) = restricted_spectrum(&a, &c, 2, 1, p, 40).unwrap();
        assert!(Float::with_val(p, &spectrum[0] - 0.5).abs() < (Float::with_val(p, 1) >> 110_u32));
        assert_eq!(restricted_spectrum(&a, &c, 2, 0, p, 40).unwrap().1.len(), 2);
        assert!(restricted_spectrum(&a, &c, 2, 2, p, 40)
            .unwrap()
            .1
            .is_empty());
        assert!(restricted_spectrum(&a, &c, 2, 3, p, 40).is_err());
        let identity = vec![
            Float::with_val(p, 1),
            Float::with_val(p, 0),
            Float::with_val(p, 0),
            Float::with_val(p, 1),
        ];
        assert!(restricted_spectrum(&a, &identity, 2, 1, p, 40).is_err());
    }
}
