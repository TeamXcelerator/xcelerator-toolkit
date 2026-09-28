//! Bounded-endpoint prolate Galerkin model.
//!
//! Put x=lambda*t and c=2*pi*lambda^2. In the orthonormal Legendre basis
//! phi_j=sqrt((2j+1)/2) P_j, multiplication by t couples j to j+1 with
//! a_j=(j+1)/sqrt((2j+1)(2j+3)). Thus the even block has diagonal
//! j(j+1)+c^2(a_j^2+a_(j-1)^2) and off-diagonal c^2*a_j*a_(j+1).
//! This represents the bounded, natural singular-endpoint realization.
//! Reported residuals include the one omitted Jacobi coupling. They are
//! computed diagnostics, not interval certificates of continuum accuracy.
use anyhow::{ensure, Result};
use nalgebra::{DMatrix, SymmetricEigen};

pub(super) const SEMANTICS: &str = "prolate-bounded-legendre-even-v2";
#[cfg(feature = "hp")]
pub(super) const SPECTRUM_DISCRETIZATION: &str = "prolate-bounded-legendre-even-v1";

// Charge retained output together with solver/polynomial scratch. Sampling
// evaluates at most `terms` polynomials, of degree below 2*n, at each point.
// Precision-word weighting is a work policy, not a floating-operation estimate.
pub(super) fn resource_budget(
    n: usize,
    samples: usize,
    terms: usize,
    p: u32,
    native: bool,
) -> Result<()> {
    let n = n as u128;
    let samples = samples as u128;
    let limb_bytes = u128::from(p).div_ceil(64) * 8;
    let bytes = if native {
        // Completion retains five dense matrices at its peak; allow a sixth
        // for eigensolver scratch in addition to the linear vectors below.
        48u128
            .saturating_mul(n)
            .saturating_mul(n)
            .saturating_add(256 * n)
            .saturating_add(16 * samples)
            .saturating_add(4096)
    } else {
        // Float storage and allocator overhead; include p+64 work temporaries.
        let work_bytes = u128::from(p + 64).div_ceil(64) * 8 + 96;
        32 * n * work_bytes + 2 * samples * (limb_bytes + 96) + 64 * work_bytes
    };
    ensure!(
        bytes <= 1u128 << 30,
        "prolate combined output and numerical workspace exceeds 1 GiB"
    );
    let words = if native {
        1
    } else {
        u128::from(p + 64).div_ceil(64)
    };
    let work = samples
        .saturating_mul(terms as u128)
        .saturating_mul(2)
        .saturating_mul(n)
        .saturating_mul(words);
    ensure!(
        work <= 1u128 << 32,
        "prolate total polynomial sampling work budget exceeded"
    );
    ensure!(
        eigen_work(n as usize, p, native) <= 1u128 << 32,
        "prolate eigensolver work budget exceeded"
    );
    Ok(())
}

fn eigen_work(n: usize, p: u32, native: bool) -> u128 {
    let size = n as u128;
    size.saturating_mul(size).saturating_mul(if native {
        size
    } else {
        u128::from(p + 64).div_ceil(64)
    })
}

fn next_size(n: usize, budget: usize) -> Result<usize> {
    ensure!(
        n < budget,
        "prolate Legendre truncation remains unresolved at the maximum resolution budget"
    );
    Ok((n + (n / 2).max(8)).min(budget))
}

fn initial_size(budget: usize) -> Result<usize> {
    ensure!(
        (16..=100_000).contains(&budget),
        "prolate resolution budget must be in 16..=100000 Legendre coefficients"
    );
    Ok(24.min(budget))
}

fn block_f64(c: f64, n: usize) -> (Vec<f64>, Vec<f64>) {
    let c2 = c * c;
    let mut diagonal = Vec::with_capacity(n);
    let mut off = Vec::with_capacity(n);
    for i in 0..n {
        let j = (2 * i) as f64;
        diagonal.push(
            j * (j + 1.0)
                + c2 * (2.0 * j * j + 2.0 * j - 1.0) / ((2.0 * j - 1.0) * (2.0 * j + 3.0)),
        );
        off.push(
            c2 * (j + 1.0) * (j + 2.0)
                / ((2.0 * j + 3.0) * ((2.0 * j + 1.0) * (2.0 * j + 5.0)).sqrt()),
        );
    }
    (diagonal, off)
}

fn evaluate_f64(v: &[f64], t: f64) -> f64 {
    let mut previous = 1.0;
    let mut current = t;
    let mut value = v[0] / 2.0f64.sqrt();
    for degree in 2..(2 * v.len()) {
        let polynomial = ((2 * degree - 1) as f64 * t * current - (degree - 1) as f64 * previous)
            / degree as f64;
        if degree % 2 == 0 {
            value += v[degree / 2] * ((2 * degree + 1) as f64 / 2.0).sqrt() * polynomial;
        }
        previous = current;
        current = polynomial;
    }
    value
}

pub(super) fn compute_f64(cfg: &super::ProlateConfig) -> Result<super::ProlateResult> {
    let start = std::time::Instant::now();
    ensure!(
        cfg.precision_bits == 53
            && cfg.lambda.is_finite()
            && cfg.lambda > 1.0
            && cfg.n_sample >= 2
            && cfg.n_sample <= u32::MAX as usize,
        "invalid native prolate precision, cutoff, or sampling configuration"
    );
    let lambda = cfg.lambda;
    ensure!(
        lambda * lambda < 1_000_000.0,
        "prolate finite-sum work budget exceeded"
    );
    let c = 2.0 * std::f64::consts::PI * lambda * lambda;
    let terms = (lambda * lambda).ceil() as usize + 1;
    let mut n = initial_size(cfg.n_grid)?;
    let mut cumulative_eigen_work = 0u128;
    let (values, vectors, residual) = loop {
        cumulative_eigen_work = cumulative_eigen_work.saturating_add(eigen_work(n, 53, true));
        ensure!(
            cumulative_eigen_work <= 1u128 << 32,
            "prolate cumulative eigensolver work budget exceeded"
        );
        resource_budget(n, cfg.n_sample, terms, 53, true)?;
        let (d, b) = block_f64(c, n);
        let mut matrix = DMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            matrix[(i, i)] = d[i];
            if i + 1 < n {
                matrix[(i, i + 1)] = b[i];
                matrix[(i + 1, i)] = b[i];
            }
        }
        let mut eig = SymmetricEigen::try_new(matrix.clone(), f64::EPSILON, n.saturating_mul(128))
            .ok_or_else(|| anyhow::anyhow!("Legendre eigensolver did not converge"))?;
        let eigenvalues = xc_numerics::symmetric_f64::complete_symmetric_eigensystem_f64(
            matrix.as_slice(),
            n,
            eig.eigenvectors.as_mut_slice(),
        )?;
        let mut order = (0..n).collect::<Vec<_>>();
        order.sort_by(|&a, &b| eigenvalues[a].total_cmp(&eigenvalues[b]));
        let mut values = Vec::new();
        let mut vectors = Vec::new();
        let mut largest = 0.0f64;
        for index in [0, 2] {
            let value = eigenvalues[order[index]];
            let mut vector = eig
                .eigenvectors
                .column(order[index])
                .iter()
                .copied()
                .collect::<Vec<_>>();
            if evaluate_f64(&vector, 0.0) < 0.0 {
                for x in &mut vector {
                    *x = -*x;
                }
            }
            let mut squared = (b[n - 1] * vector[n - 1]).powi(2);
            for i in 0..n {
                let mut r = (d[i] - value) * vector[i];
                if i > 0 {
                    r += b[i - 1] * vector[i - 1];
                }
                if i + 1 < n {
                    r += b[i] * vector[i + 1];
                }
                squared += r * r;
            }
            largest = largest.max(squared.sqrt() / (1.0 + value.abs()));
            values.push(value);
            vectors.push(vector);
        }
        if largest.is_finite() && largest <= 2.0f64.powi(-39) {
            break (values, vectors, largest);
        }
        n = next_size(n, cfg.n_grid)?;
    };
    ensure!(
        vectors[0][0] != 0.0,
        "prolate ground integral is unresolved"
    );
    let c0 = -vectors[1][0] / vectors[0][0];
    let mut combination = vectors[1]
        .iter()
        .zip(&vectors[0])
        .map(|(a, b)| (a + c0 * b) / lambda.sqrt())
        .collect::<Vec<_>>();
    combination[0] = 0.0; // enforce the defining zero integral in this basis
    let mut u_grid = (0..cfg.n_sample)
        .map(|i| (lambda.ln() * (2.0 * i as f64 / (cfg.n_sample - 1) as f64 - 1.0)).exp())
        .collect::<Vec<_>>();
    u_grid[0] = 1.0 / lambda;
    u_grid[cfg.n_sample - 1] = lambda;
    // Open support is retained for the Eisenstein sum. The underlying prolate
    // polynomial is bounded and generally nonzero at t=+/-1.
    // The represented grid and cutoff can straddle the same exact jump.
    // Resolve the rounding band as equality under the open-support rule.
    let support_upper = lambda * (1.0 - 2.0f64.powi(-45));
    let k_values = u_grid
        .iter()
        .map(|&u| {
            let mut sum = 0.0;
            let bound = (lambda / u).ceil() as usize;
            for k in 1..=bound {
                let x = k as f64 * u;
                if x >= support_upper {
                    break;
                }
                sum += evaluate_f64(&combination, x / lambda);
            }
            u.sqrt() * sum
        })
        .collect::<Vec<_>>();
    ensure!(
        k_values.iter().all(|x| x.is_finite()) && c0.is_finite(),
        "nonfinite prolate polynomial sampling"
    );
    Ok(super::ProlateResult {
        k_values,
        u_grid,
        eigenvalue_0: values[0],
        eigenvalue_4: values[1],
        c_4: 1.0,
        c_0: c0,
        elapsed_seconds: start.elapsed().as_secs_f64(),
        discretization: SEMANTICS.into(),
        resolution_budget: cfg.n_grid,
        basis_dimension: n,
        relative_operator_residual: Some(residual),
    })
}

#[cfg(feature = "hp")]
pub(super) mod hp {
    use super::*;
    use rug::Float;
    use xc_numerics::eigen::{
        tridiag_eigenvector_for_value_hp, tridiag_sturm_count_below_hp, TridiagEigvecOptions,
    };

    pub(crate) fn block(lambda: &Float, n: usize, p: u32) -> (Vec<Float>, Vec<Float>) {
        let mut c = Float::with_val(p, rug::float::Constant::Pi);
        c *= 2;
        c *= lambda;
        c *= lambda;
        c.square_mut();
        let mut d = Vec::with_capacity(n);
        let mut b = Vec::with_capacity(n);
        for i in 0..n {
            let j = (2 * i) as i64;
            let mut entry = Float::with_val(p, 2 * j * j + 2 * j - 1);
            entry /= (2 * j - 1) * (2 * j + 3);
            entry *= &c;
            entry += j * (j + 1);
            d.push(entry);
            let mut off = Float::with_val(p, (j + 1) * (j + 2));
            off *= &c;
            let mut denominator = Float::with_val(p, (2 * j + 1) * (2 * j + 5)).sqrt();
            denominator *= 2 * j + 3;
            off /= denominator;
            b.push(off);
        }
        (d, b)
    }

    pub(crate) fn validate_spectrum(
        d: &[Float],
        b: &[Float],
        values: &[Float],
        p: u32,
    ) -> Result<()> {
        ensure!(
            values.len() == d.len() && b.len() + 1 == d.len(),
            "prolate spectrum shape mismatch"
        );
        let work = p + 64;
        let norm = (0..d.len())
            .map(|i| {
                let mut row = Float::with_val(work, &d[i]).abs();
                if i > 0 {
                    row += Float::with_val(work, &b[i - 1]).abs();
                }
                if i < b.len() {
                    row += Float::with_val(work, &b[i]).abs();
                }
                row
            })
            .max_by(Float::total_cmp)
            .ok_or_else(|| anyhow::anyhow!("empty prolate matrix"))?;
        let radius = (norm * (8 * d.len())) >> p;
        for (i, value) in values.iter().enumerate() {
            ensure!(
                value.is_finite(),
                "prolate spectrum contains nonfinite values"
            );
            let lower = Float::with_val(work, value - &radius);
            let upper = Float::with_val(work, value + &radius);
            let below_lower = tridiag_sturm_count_below_hp(d, b, &lower, work)?;
            let below_upper = tridiag_sturm_count_below_hp(d, b, &upper, work)?;
            // The interval must contain the value at this algebraic index.
            // It need not isolate a simple eigenvalue: parity pairs of a finite
            // Dirichlet grid can be closer than the source rounding scale.
            ensure!(
                below_lower <= i && below_upper > i,
                "prolate cached eigenvalue does not enclose its ordered source-matrix index: index={i}, counts=[{below_lower},{below_upper}]"
            );
        }
        Ok(())
    }

    fn evaluate(v: &[Float], t: &Float, p: u32) -> Float {
        let mut previous = Float::with_val(p, 1);
        let mut current = Float::with_val(p, t);
        let mut value = Float::with_val(p, &v[0]) / Float::with_val(p, 2).sqrt();
        for degree in 2..2 * v.len() {
            let mut polynomial = Float::with_val(p, &current * t);
            polynomial *= 2 * degree - 1;
            polynomial -= Float::with_val(p, &previous * (degree - 1));
            polynomial /= degree;
            if degree % 2 == 0 {
                let mut scale = Float::with_val(p, 2 * degree + 1);
                scale /= 2;
                scale.sqrt_mut();
                scale *= &v[degree / 2];
                scale *= &polynomial;
                value += scale;
            }
            previous = current;
            current = polynomial;
        }
        value
    }

    pub(crate) fn compute<F>(
        lambda: &Float,
        budget: usize,
        samples: usize,
        p: u32,
        mut spectrum: F,
    ) -> Result<super::super::hp::HpProlateResult>
    where
        F: FnMut(&[Float], &[Float], u32) -> Result<Vec<Float>>,
    {
        let start = std::time::Instant::now();
        ensure!(
            (32..=1_000_000).contains(&p)
                && lambda.is_finite()
                && lambda > &1
                && samples >= 2
                && samples <= u32::MAX as usize,
            "invalid HP prolate precision, cutoff, or sampling configuration"
        );
        let lambda = Float::with_val(p, lambda);
        ensure!(
            lambda > 1 && lambda.clone().square() < 1_000_000,
            "prolate finite-sum work budget exceeded"
        );
        let work = p + 64;
        // The pinned lower sample is RN(1/lambda). One extra term covers its
        // rounding relative to lambda^2 throughout the admitted precision range.
        let terms = (Float::with_val(work, &lambda) * &lambda)
            .ceil()
            .to_integer()
            .and_then(|x| x.to_usize())
            .ok_or_else(|| anyhow::anyhow!("prolate sampling work bound is unrepresentable"))?
            + 1;
        let mut n = initial_size(budget)?;
        let tolerance = Float::with_val(work, 1) >> p.saturating_sub(12);
        let mut cumulative_eigen_work = 0u128;
        let (values, vectors, residual) = loop {
            cumulative_eigen_work = cumulative_eigen_work.saturating_add(eigen_work(n, p, false));
            ensure!(
                cumulative_eigen_work <= 1u128 << 32,
                "prolate cumulative eigensolver work budget exceeded"
            );
            resource_budget(n, samples, terms, p, false)?;
            let (d, b) = block(&lambda, n, work);
            let eigenvalues = spectrum(&d, &b[..n - 1], work)?;
            validate_spectrum(&d, &b[..n - 1], &eigenvalues, work)?;
            let mut values = Vec::new();
            let mut vectors = Vec::new();
            let mut largest = Float::with_val(work, 0);
            for index in [0, 2] {
                let value = eigenvalues[index].clone();
                let mut vector = tridiag_eigenvector_for_value_hp(
                    &d,
                    &b[..n - 1],
                    &value,
                    work,
                    TridiagEigvecOptions::default(),
                )?;
                if evaluate(&vector, &Float::with_val(work, 0), work) < 0 {
                    for x in &mut vector {
                        *x = -x.clone();
                    }
                }
                let mut squared = Float::with_val(work, &b[n - 1] * &vector[n - 1]).square();
                for i in 0..n {
                    let mut r = Float::with_val(work, &d[i] - &value);
                    r *= &vector[i];
                    if i > 0 {
                        r += Float::with_val(work, &b[i - 1] * &vector[i - 1]);
                    }
                    if i + 1 < n {
                        r += Float::with_val(work, &b[i] * &vector[i + 1]);
                    }
                    squared += r.square();
                }
                let relative = squared.sqrt() / (Float::with_val(work, &value).abs() + 1);
                if relative > largest {
                    largest = relative;
                }
                values.push(value);
                vectors.push(vector);
            }
            if largest.is_finite() && largest <= tolerance {
                break (values, vectors, largest);
            }
            n = next_size(n, budget)?;
        };
        ensure!(
            !vectors[0][0].is_zero(),
            "prolate ground integral is unresolved"
        );
        let c0 = -Float::with_val(work, &vectors[1][0]) / &vectors[0][0];
        let sqrt_lambda = Float::with_val(work, &lambda).sqrt();
        let mut combination = vectors[1]
            .iter()
            .zip(&vectors[0])
            .map(|(a, b)| {
                let mut x = Float::with_val(work, &c0 * b);
                x += a;
                x / &sqrt_lambda
            })
            .collect::<Vec<_>>();
        combination[0] = Float::with_val(work, 0);
        let logarithm = Float::with_val(work, &lambda).ln();
        let mut u_grid = (0..samples)
            .map(|i| {
                let mut x = Float::with_val(work, 2 * i);
                x /= samples - 1;
                x -= 1;
                x *= &logarithm;
                Float::with_val(p, x.exp())
            })
            .collect::<Vec<_>>();
        u_grid[0] = Float::with_val(p, 1) / &lambda;
        u_grid[samples - 1] = lambda.clone();
        // Match exact endpoint/interior coincidences despite p-bit grid rounding.
        let support_upper =
            Float::with_val(work, &lambda) - (Float::with_val(work, &lambda) >> (p - 8));
        let k_values = u_grid
            .iter()
            .map(|u| -> Result<Float> {
                let bound = (Float::with_val(work, &lambda) / u)
                    .ceil()
                    .to_integer()
                    .and_then(|x| x.to_usize())
                    .ok_or_else(|| {
                        anyhow::anyhow!("prolate sample work bound is unrepresentable")
                    })?;
                ensure!(bound <= 1_000_000, "prolate sample work budget exceeded");
                let mut sum = Float::with_val(work, 0);
                for k in 1..=bound {
                    let x = Float::with_val(work, u) * k;
                    if x >= support_upper {
                        break;
                    }
                    sum += evaluate(&combination, &(x / &lambda), work);
                }
                sum *= Float::with_val(work, u).sqrt();
                ensure!(sum.is_finite(), "prolate sample exceeds exponent range");
                Ok(Float::with_val(p, sum))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(super::super::hp::HpProlateResult {
            k_values,
            u_grid,
            eigenvalue_0: Float::with_val(p, &values[0]),
            eigenvalue_4: Float::with_val(p, &values[1]),
            c_4: Float::with_val(p, 1),
            c_0: Float::with_val(p, c0),
            elapsed_seconds: start.elapsed().as_secs_f64(),
            precision_bits: p,
            discretization: SEMANTICS.into(),
            resolution_budget: budget,
            basis_dimension: n,
            relative_operator_residual: Some(Float::with_val(p, residual)),
        })
    }
}

#[cfg(test)]
mod resource_tests {
    #[test]
    fn resource_budget_covers_output_and_total_sampling() {
        super::resource_budget(81, 256, 14, 53, true).unwrap();
        super::resource_budget(512, 256, 14, 3386, false).unwrap();
        assert!(super::resource_budget(24, u32::MAX as usize, 2, 53, true).is_err());
        assert!(super::resource_budget(24, u32::MAX as usize, 2, 128, false).is_err());
        assert!(super::resource_budget(24, 256, 1_000_000, 53, true).is_err());
        assert!(super::resource_budget(512, 1_000_000, 14, 3386, false).is_err());
    }
}
