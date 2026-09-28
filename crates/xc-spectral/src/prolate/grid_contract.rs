//! Contracts for a canonical uniform Dirichlet grid and sampled node counts.
#[cfg(feature = "hp")]
use anyhow::{ensure, Result};
#[cfg(feature = "hp")]
use rug::Float;

pub(super) fn nodes_f64(values: &[f64]) -> usize {
    let scale = values.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
    if scale == 0.0 || values.iter().any(|x| !x.is_finite()) {
        return 0;
    }
    let mut previous = None;
    let mut count = 0;
    for value in values {
        if *value == 0.0 || (value / scale).abs() < super::NODE_NOISE_FACTOR {
            continue;
        }
        let sign = value.is_sign_positive();
        count += usize::from(previous.is_some_and(|old| old != sign));
        previous = Some(sign);
    }
    count
}

#[cfg(feature = "hp")]
pub(super) fn nodes_hp(values: &[Float], p: u32) -> Result<usize> {
    ensure!(
        (32..=1_000_000).contains(&p),
        "node-count precision must be in 32..=1000000 bits"
    );
    ensure!(
        values.iter().all(Float::is_finite),
        "node-count samples must be finite"
    );
    let scale = values
        .iter()
        .map(|x| x.clone().abs())
        .max_by(|a, b| a.partial_cmp(b).unwrap());
    let Some(scale) = scale.filter(|x| !x.is_zero()) else {
        return Ok(0);
    };
    let tolerance = Float::with_val(p, Float::parse("1e-6").unwrap());
    let mut previous = None;
    let mut count = 0;
    for value in values {
        if value.is_zero() || Float::with_val(p, value / &scale).abs() < tolerance {
            continue;
        }
        let sign = value.is_sign_positive();
        count += usize::from(previous.is_some_and(|old| old != sign));
        previous = Some(sign);
    }
    Ok(count)
}

#[cfg(feature = "hp")]
pub(super) fn validate_grid(values: &[Float], lambda: &Float, h: &Float, p: u32) -> Result<()> {
    ensure!(
        (32..=1_000_000).contains(&p),
        "interpolation precision must be in 32..=1000000 bits"
    );
    ensure!(
        !values.is_empty() && values.len() < u32::MAX as usize,
        "interpolation needs a nonempty grid below u32::MAX"
    );
    ensure!(
        values.iter().all(Float::is_finite),
        "interpolation samples must be finite"
    );
    ensure!(
        lambda.is_finite() && lambda > &0 && h.is_finite() && h > &0,
        "interpolation lambda and spacing must be finite and positive"
    );
    let mut expected = Float::with_val(p, lambda);
    expected *= 2u32;
    if expected.is_finite() {
        expected /= values.len() + 1;
    } else {
        expected = Float::with_val(p, lambda);
        expected /= values.len() + 1;
        expected *= 2u32;
    }
    ensure!(
        expected > 0 && expected.is_finite() && Float::with_val(p, h) == expected,
        "interpolation spacing must equal 2*lambda/(N+1) rounded to working precision"
    );
    Ok(())
}

/// Caller has validated the canonical grid and the finite evaluation point.
/// Nearest-endpoint distances preserve boundary values and avoid overflowing x+lambda.
#[cfg(feature = "hp")]
pub(super) fn interpolate(values: &[Float], lambda: &Float, x: &Float, p: u32) -> Float {
    if x.clone().abs() >= *lambda {
        return Float::with_val(p, 0);
    }
    let n = values.len();
    let from_right = x >= &0;
    // Retain a small nonzero boundary distance before normalizing it.
    let mut position = if from_right {
        Float::with_val(p, lambda - x)
    } else {
        Float::with_val(p, lambda + x)
    };
    position /= lambda;
    position *= n + 1;
    position /= 2u32;
    let lower = position
        .clone()
        .floor()
        .to_integer()
        .unwrap()
        .to_usize()
        .unwrap();
    let fraction = position - lower;
    let sample = |index: usize| {
        if index == 0 || index > n {
            Float::with_val(p, 0)
        } else if from_right {
            Float::with_val(p, &values[n - index])
        } else {
            Float::with_val(p, &values[index - 1])
        }
    };
    let mut result = Float::with_val(p, 1);
    result -= &fraction;
    result *= sample(lower);
    result += Float::with_val(p, fraction * sample(lower + 1));
    result
}

#[cfg(test)]
mod tests {
    #[test]
    fn native_node_counts_are_scale_invariant_and_skip_exact_zeros() {
        for scale in [1e-200, 1., 1e200, f64::from_bits(1)] {
            assert_eq!(super::nodes_f64(&[scale, 0., -scale, 0., scale]), 2);
        }
    }
    #[test]
    fn exhaustive_prolate_native_interpolation_preserves_boundary_distance() {
        let x = f64::from_bits(1.0_f64.to_bits() - 1);
        let values = [1.0, 1.0, 1.0];
        let expected = 2.0 * (1.0 - x);
        for point in [x, -x] {
            assert_eq!(
                crate::prolate::interp_grid_f64(&values, 1.0, 0.5, point),
                expected
            );
        }
    }
    #[cfg(feature = "hp")]
    #[test]
    fn exhaustive_prolate_hp_interpolation_preserves_boundary_distance() {
        use rug::Float;
        let p = 128;
        let lambda = Float::with_val(p, 1);
        let mut x = lambda.clone();
        x.next_down();
        let values = vec![Float::with_val(p, 1); 3];
        let expected = Float::with_val(p, &lambda - &x) * 2u32;
        for point in [x.clone(), -x] {
            let got = crate::prolate::hp::try_interp_grid(
                &values,
                &lambda,
                &Float::with_val(p, 0.5),
                &point,
                p,
            )
            .unwrap();
            assert_eq!(got, expected);
        }
    }
}
