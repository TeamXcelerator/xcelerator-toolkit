//! Independent identities: logistic primitive, prime derivative jumps, and
//! finite-difference energy from edge fluxes rather than matrix entries.
use xc_spectral::{mellin, prolate};

#[test]
fn mellin_at_zero_matches_logistic_primitive() {
    for lambda in [1.0001_f64, 2.0, 5.0, 10.0] {
        let expected = 1.0 / (1.0 + (1.0 / lambda).exp()) - 1.0 / (1.0 + lambda.exp());
        let (actual, imaginary) = mellin::try_truncated_lambda_f64(0.0, 0.0, lambda, 96).unwrap();
        assert!((actual - expected).abs() < 8.0e-15 * expected.abs().max(0.01));
        assert_eq!(imaginary, 0.0);
    }
}

#[test]
fn prolate_quadratic_form_matches_independent_edge_energy() {
    for n in [3_usize, 7, 15] {
        for lambda in [0.25, 1.0, 3.0] {
            let cfg = prolate::ProlateConfig::new(lambda, n);
            let (diagonal, off) = prolate::try_build_pw_matrix_f64(&cfg).unwrap();
            let h = 2.0 * lambda / (n + 1) as f64;
            let v: Vec<f64> = (0..n).map(|j| ((j * 7 + 3) % 11) as f64 - 5.0).collect();
            let mut energy = 0.0;
            for edge in 0..=n {
                let x = -lambda + (edge as f64 + 0.5) * h;
                let left = if edge == 0 { 0.0 } else { v[edge - 1] };
                let right = if edge == n { 0.0 } else { v[edge] };
                energy += (lambda * lambda - x * x) * ((right - left) / h).powi(2);
            }
            for (j, value) in v.iter().enumerate() {
                let x = -lambda + (j + 1) as f64 * h;
                energy += (2.0 * std::f64::consts::PI * lambda * x * value).powi(2);
            }
            let matrix_energy = (0..n).map(|j| diagonal[j] * v[j] * v[j]).sum::<f64>()
                + 2.0 * (0..n - 1).map(|j| off[j] * v[j] * v[j + 1]).sum::<f64>();
            assert!((matrix_energy - energy).abs() < 2.0e-14 * energy);
        }
    }
}

#[cfg(feature = "hp")]
#[test]
fn hp_mellin_at_zero_and_constant_weight_match_closed_form() {
    use rug::Float;
    use xc_numerics::quadrature::{try_gauss_legendre_nodes, CacheMode};
    let p = 160;
    let (nodes, weights) = try_gauss_legendre_nodes(96, p, CacheMode::Off).unwrap();
    for integer in [2_u32, 5, 10] {
        let lambda = Float::with_val(p, integer);
        let zero = Float::with_val(p, 0);
        let high_lambda = Float::with_val(320, integer);
        let expected = (Float::with_val(320, high_lambda.clone().recip().exp() + 1u32)).recip()
            - (high_lambda.exp() + 1u32).recip();
        let (actual, imaginary) =
            mellin::try_truncated_lambda_hp(&zero, &zero, &lambda, &nodes, &weights).unwrap();
        assert!(Float::with_val(320, &actual - expected).abs() < (Float::with_val(320, 1) >> 130));
        assert!(imaginary.is_zero());
        let constant = (lambda.clone().ln() * 2u32).sqrt();
        let (weighted, weighted_imaginary) = mellin::try_xi_weighted_mellin_hp(
            &zero,
            &zero,
            &lambda,
            &[constant],
            0,
            &nodes,
            &weights,
        )
        .unwrap();
        assert!((weighted - actual).abs() < (Float::with_val(p, 1) >> 130));
        assert!(weighted_imaginary.is_zero());
    }
}

#[cfg(feature = "hp")]
#[test]
fn screw_prime_power_derivative_jumps_match_von_mangoldt_weight() {
    use rug::Float;
    // Suzuki (2023), Eq. 1.1 and g=-Psi: a prime-power term turns on
    // with slope log(p)/sqrt(n). Smooth terms vanish in this jump limit.
    // https://arxiv.org/pdf/2206.03682, pp. 1-2.
    for p in [96, 192] {
        let kernel = xc_spectral::screw::ScrewKernel::try_new(2.0, p).unwrap();
        let delta = Float::with_val(p, 1) >> 40;
        for (n, prime) in [(2_u32, 2_u32), (4, 2), (8, 2), (9, 3), (6, 1)] {
            let t = Float::with_val(p, n).ln();
            let mut difference = kernel.try_eval(&Float::with_val(p, &t + &delta)).unwrap();
            difference += kernel.try_eval(&Float::with_val(p, &t - &delta)).unwrap();
            difference -= kernel.try_eval(&t).unwrap() * 2u32;
            difference /= &delta;
            let expected = Float::with_val(p, prime).ln() / Float::with_val(p, n).sqrt();
            assert!(
                (difference - expected).abs() < (Float::with_val(p, 1) >> 32),
                "jump at n={n}, p={p}"
            );
        }
    }
}
